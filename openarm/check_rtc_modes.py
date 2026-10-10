"""Bounded one-load RTC comparison on the saved Pillow checkpoint, without a robot.

Native sampler comparisons restore identical RNG/input state. Diagnostic hooks
add overhead, so these timings are not a transport deadline or sustained-rate test.
Run only in the registered one-GPU zero-update inference job (1200 seconds max).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
from unittest.mock import patch

from openarm.check_inference import completed_formal_checkpoint
from openarm.check_rtc_numerics import ATOL, RTOL, difference, fingerprint, native_observation
from openarm.data import CAMERAS, GROUPS, job_guard, write_json


SEED = 20261010
PREFIX, END, HORIZON = 4, 16, 32
CALL_LIMIT = 48


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-dir", "warmup-sample", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1100)
    args = parser.parse_args(argv)
    job_guard()
    if (os.environ.get("MODEL_LAB_STAGE_KIND") != "none"
            or os.environ.get("MODEL_LAB_MAX_STEPS") != "0"
            or os.environ.get("MODEL_LAB_GPUS") != "1"):
        parser.error("RTC modes require a registered one-GPU zero-update stage")
    if not 0 < args.timeout_seconds <= 1100:
        parser.error("Timeout must be 1..1100 seconds inside the 1200-second job")
    root = Path(os.environ["RLDX_OUTPUT"]).resolve()
    args.output = args.output.resolve()
    if args.output.exists() or not args.output.is_relative_to(root):
        parser.error("Evidence must be a fresh file under private RLDX_OUTPUT")
    for name in ("checkpoint_dir", "warmup_sample"):
        path = getattr(args, name).resolve(strict=True)
        if not path.is_relative_to(root):
            parser.error("Checkpoint and sample must be under private RLDX_OUTPUT")
        setattr(args, name, path)
    completed_formal_checkpoint(args.checkpoint_dir)
    return args


def regions(left, right, prefix=PREFIX, end=END):
    """Errors in actual normalized units or radians, without quality ranking."""
    return {name: difference(left[start:stop], right[start:stop])
            for name, start, stop in (("frozen", 0, prefix),
                                     ("transition", prefix, end), ("free", end, HORIZON))
            if stop > start}


def check_native_trace(trace, mode, vjps, steps=4):
    if not isinstance(trace, dict):
        raise AssertionError("Native sampler did not return its effective branch trace")
    if (trace.get("effective_mode") != mode or trace.get("vjp_calls") != vjps
            or trace.get("denoising_steps") != steps):
        raise AssertionError(f"Requested and effective native RTC paths differ: {trace}")


def bounded_short_prior(codec, raw, current):
    """Two points inside every offset's range, sampled with terminal endpoint hold.

    Decoder ranges vary by horizon. The intersection avoids labeling native
    clipping as an exact frozen prefix; it is not an actuator safety interval.
    """
    import numpy as np
    from openarm.rtc import ExecutionRTC, INTERVAL_NS
    low = codec.decode_unclipped(np.full((1, HORIZON, codec.dim), -.75, np.float32), current)
    high = codec.decode_unclipped(np.full((1, HORIZON, codec.dim), .75, np.float32), current)
    lower, upper = low.max(axis=0), high.min(axis=0)
    if (lower > upper).any():
        raise ValueError("No common decoder interval for the short held-prior probe")
    first = np.clip(raw, lower, upper)
    second = first.copy()
    second[0] = np.clip(first[0] + .001, lower[0], upper[0])
    second[8] = np.clip(first[8] - .001, lower[8], upper[8])
    plan = {"positions": np.stack((first, second)), "start_timestamp_ns": 1,
            "interval_ns": INTERVAL_NS}
    target = ExecutionRTC.sample(plan, 1 + np.arange(HORIZON, dtype=np.int64) * INTERVAL_NS)
    if not np.array_equal(target[2:], np.repeat(second[None], HORIZON - 2, axis=0)):
        raise AssertionError("Native executor-plan sampling lost the held endpoint")
    return target


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"complete": False, "checkpoint": str(args.checkpoint_dir),
        "checkpoint_updates": 500, "scope": "offline_native_sampler_and_backend_warmup",
        "robot_execution": False, "transport_execution": False,
        "executor_timing_validated": False, "seed": SEED,
        "inference_call_limit": CALL_LIMIT, "paired_rng_states_restored": True,
        "tolerance": {"atol": ATOL, "rtol": RTOL, "units": "normalized action"},
        "latency_scope": "synchronized whole local pipeline with numerical instrumentation; no p95/p99 claim",
        "baseline_scope": "ordinary explicit bootstrap in the guided-policy no_grad context",
        "checks": [], "warmup": [], "limitations": [
            "One recorded observation; no task-quality or sustained-rate evidence",
            "Inpainting is inference-only conditioning, not trained RTC",
            "Guidance is soft; smaller prior error or latency does not prove a better policy"]}
    started, policy, error = time.monotonic(), None, None
    native_action = None
    total_calls = 0

    def timeout(signum, frame):
        raise TimeoutError("RTC mode probe exceeded its bounded timeout")
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")
        import cv2
        import numpy as np
        import pyarrow as pa
        import torch
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy
        from rldx.policy.step_request import decode_options_to_step_request
        from openarm.rtc import PriorCodec
        from openarm.serve import inspect_checkpoint

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        cv2.setNumThreads(2)
        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
        evidence["threads"] = {"torch": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
            "opencv": cv2.getNumThreads(), "arrow_cpu": pa.cpu_count(), "arrow_io": pa.io_thread_count(),
            "affinity": sorted(os.sched_getaffinity(0))}
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one assigned visible CUDA device")
        checkpoint, manifest = inspect_checkpoint(args.checkpoint_dir)
        with np.load(args.warmup_sample, allow_pickle=False) as stored:
            sample = {key: stored[key].copy() for key in stored.files}
        raw = np.asarray(sample["qpos"], dtype=np.float32)
        if raw.shape != (16,) or not np.isfinite(raw).all() or sample["prompt"].shape != ():
            raise ValueError("Recorded input needs finite qpos[16] and scalar prompt")
        for key in CAMERAS:
            frame = sample["camera_" + key]
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("Recorded cameras must be RGB uint8 HWC")
        current = raw.copy()
        current[:7] += .001
        current[8:15] -= .001
        evidence.update(input_sha256=fingerprint(sample), changed_state_sha256=fingerprint(current),
            input_sha256_by_field={key: hashlib.sha256(value.tobytes()).hexdigest()
                                   for key, value in sample.items()},
            current_state_change="+0.001rad right arm, -0.001rad left arm; grippers unchanged")
        load_start = time.monotonic()
        policy = RLDXPolicy(embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
            model_path=str(checkpoint), backbone_path=str(checkpoint / "backbone"),
            device="cuda:0", strict=True, rtc_inference_mode="guided",
            rtc_inference_delay=1, rtc_inference_exec_horizon=END)
        evidence.update(model_load_s=time.monotonic() - load_start,
                        strict_loading_info=policy.model._openarm_loading_info)
        action_model, codec = policy.model.action_model, PriorCodec(policy)
        native_action = action_model.get_action_with_features

        def counted_native(*values, **options):
            nonlocal total_calls
            if total_calls >= CALL_LIMIT:
                raise RuntimeError("Total sampler call bound, including warmup, exceeded")
            total_calls += 1
            return native_action(*values, **options)

        action_model.get_action_with_features = counted_native
        schedule = getattr(action_model, "denoising_timesteps", None)
        steps = len(schedule) if schedule is not None else action_model.num_inference_timesteps
        if steps != 4:
            raise ValueError("This approved comparison requires the checkpoint's native four denoising steps")
        default_guidance = action_model._rtc.jacobian_steps_only
        evidence["native_checkpoint"] = {"denoising_steps": steps, "timesteps": schedule,
            "guidance_steps": default_guidance, "jacobian_beta": action_model._rtc.jacobian_beta}
        absolute_prior = bounded_short_prior(codec, raw, current)
        target, mask = codec.encode(absolute_prior, current)
        codec.validate_inpaint(target, current, PREFIX, END - PREFIX)
        quantized = torch.from_numpy(target).bfloat16().float().numpy()
        evidence["prior"] = {"source": "offline two-point absolute plan with endpoint hold after step1",
            "sha256": fingerprint(absolute_prior), "held_endpoint_steps": 30,
            "normalized_shape": list(target.shape), "real_channels": int(mask[0, 0].sum()),
            "padded_channels": codec.dim - 16, "max_normalized_real": float(np.abs(target[..., :16]).max()),
            "float32_absolute_roundtrip": difference(codec.decode_unclipped(target, current), absolute_prior),
            "bf16_absolute_roundtrip": difference(codec.decode_unclipped(quantized, current), absolute_prior)}
        # Exercise the actual output decoder without invoking another sampler.
        raw_states = [{key: current[start:end].copy()[None]
                       for key, (start, end) in GROUPS.items()}]

        def decode_target(value, states=raw_states):
            parts = policy.runtime._decode(torch.from_numpy(value), states)
            return np.concatenate([parts[key][0] for key in GROUPS], axis=-1)

        evidence["prior"]["native_bf16_absolute_roundtrip"] = difference(decode_target(quantized), absolute_prior)
        bad_current = current.copy()
        bad_current[0] += 10  # Deliberately invalid rebase probe; never a robot command.
        bad, _ = codec.encode(absolute_prior, bad_current)
        bad_states = [{key: bad_current[start:end].copy()[None]
                       for key, (start, end) in GROUPS.items()}]
        rejected = None
        try:
            codec.validate_inpaint(bad, bad_current, PREFIX, END - PREFIX)
        except ValueError as exc:
            rejected = str(exc)
        if rejected is None:
            raise AssertionError("Inpainting accepted an out-of-range frozen target")
        evidence["out_of_range_rebase"] = {"inpaint_rejected": True, "reason": rejected,
            "additional_raw_state_change": "+10rad on right joint0; codec-only negative test",
            "max_normalized_frozen_real": float(np.abs(bad[:, :PREFIX, :16]).max()),
            "codec_roundtrip": difference(codec.decode_unclipped(bad, bad_current), absolute_prior),
            "native_output_clipping_error": regions(decode_target(bad, bad_states), absolute_prior),
            "scope": "inpainting rejects a lossy frozen prefix; native clipping is not an actuator bound"}
        # Per-offset relative statistics are tied to the new current state.
        stale_reference_target, _ = codec.encode(absolute_prior, raw)
        evidence["rebase"] = {"encoding_changed": difference(target, stale_reference_target),
            "correct_state_roundtrip": difference(codec.decode_unclipped(target, current), absolute_prior),
            "wrong_state_decode_error": difference(codec.decode_unclipped(target, raw), absolute_prior)}
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        processed_hash = None

        def run(label, prior, mode, guidance=0, ramp=0):
            nonlocal processed_hash
            region_end = PREFIX + ramp if mode == "inpaint" else END
            if len(evidence["checks"]) >= CALL_LIMIT:
                raise RuntimeError("RTC mode inference-call bound exceeded")
            policy.reset()
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state_all(rng[3])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            call_started = time.perf_counter()
            native = native_observation(sample, current, policy.language_key)
            policy.check_observation(native)
            request = decode_options_to_step_request(native, {"session_ids": ["offline-mode-comparison"],
                                                               "executor_rtc": prior})
            unbatched, states, collated = policy.runtime._prepare_inputs(request)
            this_hash = fingerprint(collated)
            if processed_hash is not None and this_hash != processed_hash:
                raise AssertionError("Paired preprocessing changed")
            processed_hash = this_hash
            policy.runtime._inject_rtc_prefix(request, collated, len(unbatched), None)
            preprocessed = time.perf_counter()
            row = {"label": label, "mode": mode, "finite": False,
                   "processed_input_sha256": this_hash, "vjp_trace": [],
                   "region_definition": {"prefix_end": PREFIX, "transition_end": region_end,
                       "first_transition_point_also_frozen": mode == "inpaint" and ramp > 0}}
            evidence["checks"].append(row)
            encoder_inputs, velocities, vjps = [], [], []
            native_grad = torch.autograd.grad

            def capture_encoder(module, inputs):
                encoder_inputs.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

            def capture_velocity(module, inputs, output):
                velocities.append(output[:, -HORIZON:].detach().clone())

            def capture_grad(*values, **options):
                result = native_grad(*values, **options)
                vjps.append((options["grad_outputs"].detach().clone(), result[0].detach().clone()))
                return result

            handles = [action_model.action_encoder.register_forward_pre_hook(capture_encoder),
                       action_model.action_decoder.register_forward_hook(capture_velocity)]
            try:
                with patch.object(torch.autograd, "grad", side_effect=capture_grad):
                    prediction, _ = policy.runtime._run_inference(request, collated, len(unbatched), None)
                torch.cuda.synchronize()
                native_finished = time.perf_counter()
                normalized = prediction["action_pred"].detach().float()
                actions = policy.runtime._decode(normalized, states)
                decoded = np.concatenate([actions[key][0] for key in GROUPS], axis=-1)
                torch.cuda.synchronize()
                pipeline_finished = time.perf_counter()
            finally:
                for handle in handles:
                    handle.remove()
                row.update(wall_until_completion_ms=(time.perf_counter() - call_started) * 1000,
                    observed_vjp_calls=len(vjps), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
            check_native_trace(prediction.get("rtc_trace"), mode, guidance)
            trace = prediction["rtc_trace"]
            if (trace["prefix_len"] != (0 if mode == "none" else PREFIX)
                    or trace["ramp_steps"] != ramp or trace["guidance_steps"] != guidance):
                raise AssertionError("Native conditioning settings differ from the requested probe")
            if len(vjps) != guidance or len(encoder_inputs) != steps or len(velocities) != steps:
                raise AssertionError("Native step or VJP hook count differs from the requested path")
            values = normalized.cpu().numpy()[0]
            if values.shape != (HORIZON, codec.dim) or decoded.shape != (HORIZON, 16):
                raise AssertionError("Native prediction shape changed")
            if not np.isfinite(values).all() or not np.isfinite(decoded).all():
                raise AssertionError("Nonfinite native prediction")
            for residual, vjp in vjps:
                if not torch.isfinite(residual).all() or not torch.isfinite(vjp).all():
                    raise AssertionError("Nonfinite native residual or VJP")
                row["vjp_trace"].append({"residual_max": float(residual.abs().max()),
                    "padded_residual_max": float(residual[..., 16:].abs().max()),
                    "vjp_max": float(vjp.abs().max())})
            if any(item["padded_residual_max"] != 0 for item in row["vjp_trace"]):
                raise AssertionError("Guidance used padded action channels")
            if mode == "guided":
                if np.asarray(prior["mask"]).any():
                    if not any(item["residual_max"] > 0 and item["vjp_max"] > 0 for item in row["vjp_trace"]):
                        raise AssertionError("Guidance did not exercise a nonzero residual and VJP")
                elif any(item["residual_max"] != 0 or item["vjp_max"] != 0 for item in row["vjp_trace"]):
                    raise AssertionError("Zero-mask guidance produced a nonzero residual or VJP")
            trajectories = [x.float().cpu().numpy()[0] for x, _ in encoder_inputs] + [values]
            if mode == "inpaint":
                valid = np.asarray(prior["mask"])[0]
                reference = torch.as_tensor(prior["target"]).bfloat16().float().numpy()[0]
                frozen_end = PREFIX + int(ramp > 0)
                for trajectory in trajectories:
                    if not np.array_equal(trajectory[:frozen_end][valid[:frozen_end]], reference[:frozen_end][valid[:frozen_end]]):
                        raise AssertionError("Inpaint moved a masked normalized frozen-prefix value")
                # Check actual Euler updates in every region, including free padded channels.
                x = encoder_inputs[0][0]
                valid_tensor = torch.as_tensor(valid, device=x.device)
                strength = torch.ones((HORIZON, codec.dim), device=x.device, dtype=x.dtype)
                strength[:PREFIX] = torch.where(valid_tensor[:PREFIX], 0, 1)
                if ramp:
                    strength[PREFIX:PREFIX+ramp] = torch.where(valid_tensor[PREFIX:PREFIX+ramp],
                        torch.linspace(0, 1, ramp + 1, device=x.device, dtype=x.dtype)[:-1, None], 1)
                update_errors = []
                for index, velocity in enumerate(velocities):
                    x = encoder_inputs[index][0]
                    dt = prediction["rtc_trace"]["timesteps"][index + 1] - prediction["rtc_trace"]["timesteps"][index]
                    expected = (x + dt * velocity * strength).float().cpu().numpy()[0]
                    update_errors.append(difference(expected, trajectories[index + 1]))
                    if not np.allclose(expected, trajectories[index + 1], atol=ATOL, rtol=RTOL):
                        raise AssertionError("Inpaint Euler update does not match frozen/ramp/free strengths")
                row["inpaint_euler_update_errors"] = update_errors
            row.update(finite=True, normalized_shape=list(values.shape), decoded_shape=list(decoded.shape),
                native_trace=prediction["rtc_trace"], normalized_sha256=fingerprint(values), decoded_sha256=fingerprint(decoded),
                preprocess_ms=(preprocessed-call_started)*1000, native_ms=(native_finished-preprocessed)*1000,
                decode_ms=(pipeline_finished-native_finished)*1000, pipeline_ms=(pipeline_finished-call_started)*1000,
                per_step_normalized_prior_error=[regions(x[:, :16], quantized[0, :, :16], end=region_end) for x in trajectories],
                token_time_ranges=[{"min": float(t.min()), "max": float(t.max())} for _, t in encoder_inputs],
                normalized_prior_error=regions(values[:, :16], quantized[0, :, :16], end=region_end),
                decoded_absolute_prior_error=regions(decoded, absolute_prior, end=region_end),
                decoded_vs_unclipped=regions(decoded, codec.decode_unclipped(values[None], current), end=region_end))
            if mode == "inpaint":
                for index, (_, token_time) in enumerate(encoder_inputs):
                    tau = prediction["rtc_trace"]["timesteps"][index]
                    if not torch.all(token_time == tau):
                        raise AssertionError("Inference-only inpaint changed prefix token time")
                if np.asarray(prior["mask"])[:, :frozen_end, :16].all():
                    expected_absolute = decode_target(quantized)[:frozen_end]
                    if not np.allclose(decoded[:frozen_end], expected_absolute, atol=2e-5, rtol=2e-5):
                        raise AssertionError("Native decoded frozen prefix differs from quantized prior")
            return values, decoded

        def prior(mode, guidance=None, ramp=0, selected_mask=mask, selected_target=target):
            return {"mode": mode, "guidance_steps": guidance, "ramp_steps": ramp,
                    "target": selected_target, "mask": selected_mask, "prefix_len": PREFIX}

        results = {}
        settings = [("ordinary", "none", 0, 0), ("inpaint", "inpaint", 0, 0),
                    ("inpaint_ramp12", "inpaint", 0, 12)] + [
                    (f"guided{n}", "guided", n, 0) for n in (1, 3, 4)]
        for name, mode, guidance, ramp in settings:
            region_end = PREFIX + ramp if mode == "inpaint" else END
            payload = None if mode == "none" else prior(mode, guidance or None, ramp)
            first = run(name + "_first_measured", payload, mode, guidance, ramp)
            repeat = run(name + "_repeat", payload, mode, guidance, ramp)
            if not np.allclose(first[0], repeat[0], atol=ATOL, rtol=RTOL):
                raise AssertionError(f"{name} repeat exceeded the fixed tolerance")
            results[name] = first
            evidence.setdefault("comparisons", {})[name] = {
                "repeat_normalized": difference(first[0], repeat[0]),
                "repeat_decoded": difference(first[1], repeat[1]),
                "vs_ordinary_normalized": regions(first[0][:, :16], results["ordinary"][0][:, :16], end=region_end),
                "vs_ordinary_decoded": regions(first[1], results["ordinary"][1], end=region_end)}
            evidence.setdefault("latency_samples", {})[name] = {
                "count": 2, "pipeline_ms": [row["pipeline_ms"] for row in evidence["checks"][-2:]],
                "scope": "first measured and repeat for this setting; earlier settings may have warmed shared kernels"}
        default_count = steps if default_guidance is None else min(default_guidance, steps)
        default_result = run("guided_checkpoint_default", prior("guided"), "guided", default_count)
        evidence["native_default_vs_explicit"] = difference(default_result[0], results[f"guided{default_count}"][0])
        if not np.allclose(default_result[0], results[f"guided{default_count}"][0], atol=ATOL, rtol=RTOL):
            raise AssertionError("Explicit guidance-step count changed the native default path")
        for name, mode, guidance, ramp in (("inpaint_ramp12", "inpaint", 0, 12), ("guided3", "guided", 3, 0)):
            zero = run(name + "_zero_mask", prior(mode, guidance or None, ramp, np.zeros_like(mask)), mode, guidance, ramp)
            changed_target = target.copy()
            changed_target[..., 16:] = 777
            padded = run(name + "_padded_target_changed", prior(mode, guidance or None, ramp, mask, changed_target), mode, guidance, ramp)
            if (not np.allclose(zero[0], results["ordinary"][0], atol=ATOL, rtol=RTOL)
                    or not np.allclose(padded[0], results[name][0], atol=ATOL, rtol=RTOL)):
                raise AssertionError("Zero-mask invariance or padded-target independence failed")
            evidence["comparisons"][name].update(zero_mask_vs_ordinary=difference(zero[0], results["ordinary"][0]),
                padded_target_independence=difference(padded[0], results[name][0]),
                guidance_effect_same_path=difference(padded[0][:, :16], zero[0][:, :16]))
            if mode == "guided":
                effect = difference(results[name][0][:, :16], zero[0][:, :16])["max_absolute"]
                floor = evidence["comparisons"][name]["repeat_normalized"]["max_absolute"]
                evidence["comparisons"][name].update(effect_above_repeat_floor=effect > floor,
                                                     repeat_floor=floor)
                if effect <= floor:
                    raise AssertionError("Real-mask guidance effect did not exceed observed repeat noise")
        # Use the production Backend warmup, not a substitute local warming loop.
        # These predictions never enter an executor; each setting gets fresh
        # episode state and an independent calibration map on the same model.
        from openarm.backend import OpenArmBackend
        from openarm.rtc import ExecutionRTC
        for name, mode, guidance, ramp in settings[1:]:
            rtc = ExecutionRTC(policy, mode=mode, guidance_steps=guidance or None,
                               inpaint_ramp_steps=ramp, margin_ms=10, max_lateness_ms=50)
            backend = OpenArmBackend(policy, manifest, rtc=rtc,
                warmup_sample=args.warmup_sample, warmup_steps=2)
            before = total_calls
            warmup_started = time.perf_counter()
            backend.warmup()
            torch.cuda.synchronize()
            rows = backend.warmup_evidence
            evidence["warmup"].append({"setting": name, "calls": total_calls-before,
                "pipeline_ms": (time.perf_counter()-warmup_started)*1000, "records": rows,
                "latencies_ns": dict(rtc.latencies_ns)})
            modes = [item["effective_mode"] for item in rows]
            if modes.count("none") != 2 or modes.count(mode) != 2 or len(rows) != 4:
                raise AssertionError("Warmup did not execute two bootstrap and two active-mode calls")
            for item in rows:
                expected = guidance if item["effective_mode"] == "guided" else 0
                if item["vjp_calls"] != expected or item["denoising_steps"] != 4:
                    raise AssertionError("Warmup VJP/denoising counts do not match effective path")
            if rtc.context is not None or rtc.plan is not None or rtc.previous_plan is not None:
                raise AssertionError("Warmup retained synthetic context or prior plans")
            if rtc.pending or rtc.acknowledged:
                raise AssertionError("Warmup retained synthetic request or acknowledgement identities")
            if not rtc.latencies_ns or any(value <= 0 for value in rtc.latencies_ns.values()):
                raise AssertionError("Warmup lost mode-specific latency calibration")
            calibration_keys = {item["calibration_key"] for item in rows}
            if len(calibration_keys) != 2 or set(rtc.latencies_ns) != calibration_keys:
                raise AssertionError("Warmup mixed ordinary and active-mode latency calibration")
            before_reset = dict(rtc.latencies_ns)
            # Do not close the shared policy between settings; reset only clears
            # synthetic state and intentionally retains this instance's calibration.
            backend.reset("offline-warmup-check")
            if dict(rtc.latencies_ns) != before_reset:
                raise AssertionError("Episode reset changed mode-specific latency calibration")
            evidence["warmup"][-1]["calibration_survived_reset"] = True
        evidence["total_sampler_calls"] = total_calls
        evidence["complete"] = True
    except Exception as exc:
        error = exc
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if policy is not None:
            if native_action is not None:
                policy.model.action_model.get_action_with_features = native_action
            policy.reset()
        evidence["total_sampler_calls"] = total_calls
        evidence["elapsed_s"] = time.monotonic() - started
        write_json(args.output, evidence)
    if error is not None:
        raise error
    print(json.dumps({"complete": True, "output": str(args.output), "calls": len(evidence["checks"])}))


if __name__ == "__main__":
    main()
