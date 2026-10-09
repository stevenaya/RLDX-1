"""Bounded native-sampler RTC numerics; recorded input, no executor or robot.

One discarded warmup plus five paired calls use the checkpoint's native
processor, sampler and decoder. F=1/s=16 are offline conditioning parameters,
never a measured executor delay. This probe cannot override SHM timing skips.
Run only in a registered one-GPU, zero-update model job capped at 1200 seconds.
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

from openarm.data import CAMERAS, GROUPS, job_guard, write_json
from openarm.check_inference import completed_formal_checkpoint


SEED = 20261009
PREFIX = 1
GUIDANCE_END = 16
# Fixed before execution: a normalized-unit absolute bound plus 1.5625% relative
# error accommodates BF16 kernel rounding. Report exact errors and repeat noise;
# do not weaken this threshold after seeing a failure or claim bitwise identity.
ATOL = RTOL = 1 / 64


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-dir", "warmup-sample", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1100)
    args = parser.parse_args()
    job_guard()
    if (os.environ.get("MODEL_LAB_STAGE_KIND") != "none"
            or os.environ.get("MODEL_LAB_MAX_STEPS") != "0"
            or os.environ.get("MODEL_LAB_GPUS") != "1"):
        parser.error("Sampler numerics requires a registered one-GPU zero-update stage")
    if not 0 < args.timeout_seconds <= 1100:
        parser.error("Timeout must be in 1..1100 seconds inside a 1200-second job")
    root = Path(os.environ["RLDX_OUTPUT"]).resolve()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(root) or args.output.exists():
        parser.error("Evidence must be a fresh file under private RLDX_OUTPUT")
    for key in ("checkpoint_dir", "warmup_sample"):
        path = getattr(args, key).resolve(strict=True)
        if not path.is_relative_to(root):
            parser.error("Checkpoint and recorded input must be under private RLDX_OUTPUT")
        setattr(args, key, path)
    completed_formal_checkpoint(args.checkpoint_dir)
    return args


def fingerprint(value):
    """Hash processed inputs without recording private arrays or prompt text."""
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict) or hasattr(item, "items"):
            for key in sorted(item):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(len(item)).encode())
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())
    update(value)
    return digest.hexdigest()


def native_observation(sample, current, language_key):
    return {
        "video": {key: sample["camera_" + key].copy()[None, None] for key in CAMERAS},
        "state": {key: current[start:end].copy()[None, None]
                  for key, (start, end) in GROUPS.items()},
        "language": {language_key: [[str(sample["prompt"].item())]]},
    }


def difference(left, right):
    import numpy as np
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {"max_absolute": float(np.abs(delta).max()),
            "rms": float(np.sqrt(np.mean(delta ** 2)))}


def region_differences(left, right):
    return {name: difference(left[start:end], right[start:end])
            for name, start, end in (("prefix", 0, PREFIX),
                                     ("ramp", PREFIX, GUIDANCE_END),
                                     ("free", GUIDANCE_END, 32))}


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"complete": False, "checkpoint": str(args.checkpoint_dir),
        "checkpoint_updates": 500, "scope": "native_sampler_only_offline_conditioning",
        "robot_execution": False, "transport_execution": False,
        "executor_timing_validated": False, "prefix_steps": PREFIX,
        "guidance_end": GUIDANCE_END, "seed": SEED, "inference_call_limit": 6,
        "paired_rng_states_restored": True, "deterministic_algorithms_forced": False,
        "baseline_scope": "native no-prefix sampling inside the guided-policy no_grad context",
        "latency_scope": "diagnostic runtime including preprocessing, VJP tracing/synchronization and decoding",
        "tolerance": {"atol": ATOL, "rtol": RTOL, "units": "normalized action"},
        "checks": [], "limitations": ["No executor timestamps or latency feasibility claim",
            "Soft guidance does not guarantee a frozen prefix or lower final prefix error",
            "Native output decoding still clips; task quality is untested"]}
    started = time.monotonic()
    policy = None
    error = None

    def timeout(signum, frame):
        raise TimeoutError("Native RTC numerical probe exceeded its bounded timeout")
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    try:
        os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")
        import numpy as np
        import torch
        from rldx.data.embodiment_tags import EmbodimentTag
        from rldx.policy.rldx_policy import RLDXPolicy
        from rldx.policy.step_request import decode_options_to_step_request
        from openarm.backend import validate_policy
        from openarm.rtc import ExecutionRTC
        from openarm.serve import inspect_checkpoint

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one assigned visible CUDA device")
        checkpoint, _ = inspect_checkpoint(args.checkpoint_dir)
        with np.load(args.warmup_sample, allow_pickle=False) as stored:
            sample = {key: stored[key].copy() for key in stored.files}
        raw = np.asarray(sample["qpos"], dtype=np.float32)
        if raw.shape != (16,) or not np.isfinite(raw).all():
            raise ValueError("Recorded state must be finite [16]")
        if "prompt" not in sample or sample["prompt"].shape != ():
            raise ValueError("Recorded input requires a scalar prompt")
        for key in CAMERAS:
            frame = sample["camera_" + key]
            if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
                raise ValueError("Recorded cameras must be RGB uint8 HWC")
        current = raw.copy()
        current[:7] += .001
        current[8:15] -= .001
        evidence.update(input_sha256=fingerprint(sample), changed_state_sha256=fingerprint(current),
                        current_state_change="+0.001rad right arm, -0.001rad left arm; grippers unchanged")
        policy = RLDXPolicy(embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
            model_path=str(checkpoint), backbone_path=str(checkpoint / "backbone"),
            device="cuda:0", strict=True, rtc_inference_mode="guided",
            rtc_inference_delay=PREFIX, rtc_inference_exec_horizon=GUIDANCE_END)
        rtc = ExecutionRTC(policy)  # Contract + codec only; prepare/timing is never called.
        validate_policy(policy, rtc)
        evidence["strict_loading_info"] = policy.model._openarm_loading_info
        action_model = policy.model.action_model
        schedule = getattr(action_model, "denoising_timesteps", None)
        steps = len(schedule) if schedule is not None else action_model.num_inference_timesteps
        guided_steps = action_model._rtc.jacobian_steps_only
        expected_vjps = steps if guided_steps is None else min(steps, guided_steps)
        if expected_vjps < 1:
            raise ValueError("Checkpoint has no enabled Jacobian guidance steps")
        evidence["native_sampler"] = {"denoising_steps": steps, "custom_schedule": schedule,
            "jacobian_steps_only": guided_steps, "expected_vjp_calls": expected_vjps,
            "jacobian_beta": action_model._rtc.jacobian_beta}

        # A fixed offline absolute target near the recorded state, not an adopted
        # executor plan. Every inference sees exactly the same changed state.
        absolute_prior = np.repeat(raw[None], 32, axis=0)
        absolute_prior[:, 0] += np.linspace(0, .01, 32, dtype=np.float32)
        absolute_prior[:, 8] -= np.linspace(0, .01, 32, dtype=np.float32)
        target, mask = rtc.codec.encode(absolute_prior, current)
        quantized = torch.from_numpy(target).bfloat16().float().numpy()
        evidence["prior"] = {"source": "recorded qpos plus offline 0..0.01rad arm-joint ramp",
            "sha256": fingerprint(absolute_prior), "normalized_shape": list(target.shape),
            "real_channels": int(mask[0, 0].sum()), "padded_channels": target.shape[-1] - 16,
            "float32_absolute_roundtrip": difference(rtc.codec.decode_unclipped(target, current), absolute_prior),
            "bf16_normalized_quantization": difference(quantized, target),
            "bf16_absolute_roundtrip": difference(rtc.codec.decode_unclipped(quantized, current), absolute_prior)}
        rng = None
        processed_hash = None

        def run(label, prior):
            nonlocal processed_hash
            if len(evidence["checks"]) >= 6:
                raise RuntimeError("Six-call numerical probe limit exceeded")
            policy.reset()
            if rng is not None:
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                torch.cuda.set_rng_state_all(rng[3])
            native = native_observation(sample, current, policy.language_key)
            policy.check_observation(native)
            request = decode_options_to_step_request(native,
                {"session_ids": ["offline-numerics"], "executor_rtc": prior})
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            call_started = time.monotonic()
            unbatched, states, collated = policy.runtime._prepare_inputs(request)
            this_hash = fingerprint(collated)
            if processed_hash is not None and this_hash != processed_hash:
                raise AssertionError("Paired inference preprocessing changed")
            processed_hash = this_hash
            active = policy.runtime._inject_rtc_prefix(request, collated, len(unbatched), None)
            if active:
                raise AssertionError("Offline explicit targets must bypass native chunk caches")
            traces = []
            row = {"label": label, "vjp_trace": traces,
                   "processed_input_sha256": this_hash, "finite": False}
            evidence["checks"].append(row)
            native_grad = torch.autograd.grad

            def trace_grad(*grad_args, **grad_kwargs):
                residual = grad_kwargs["grad_outputs"]
                result = native_grad(*grad_args, **grad_kwargs)
                if (not torch.isfinite(residual).all()
                        or not all(torch.isfinite(value).all() for value in result)):
                    raise AssertionError("Nonfinite native guidance residual or VJP")
                padded = residual[..., 16:]
                traces.append({"residual_max": float(residual.abs().max()),
                    "padded_residual_max": float(padded.abs().max()) if padded.numel() else 0.,
                    "vjp_max": float(result[0].abs().max())})
                return result

            try:
                with patch.object(torch.autograd, "grad", side_effect=trace_grad):
                    prediction, _ = policy.runtime._run_inference(request, collated, len(unbatched), None)
            finally:
                # Preserve partial VJP traces and memory evidence even on OOM.
                row.update(vjp_calls=len(traces),
                    wall_until_native_return_ms=(time.monotonic() - call_started) * 1000,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
            normalized = prediction["action_pred"].detach().float()
            actions = policy.runtime._decode(normalized, states)
            decoded = np.concatenate([actions[key][0] for key in GROUPS], axis=-1)
            torch.cuda.synchronize()
            row.update(wall_ms=(time.monotonic() - call_started) * 1000,
                normalized_shape=list(normalized.shape), decoded_shape=list(decoded.shape),
                vjp_calls=len(traces), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if normalized.shape != (1, 32, rtc.codec.dim) or decoded.shape != (32, 16):
                raise AssertionError("Native prediction shape changed")
            if not torch.isfinite(normalized).all() or not np.isfinite(decoded).all():
                raise AssertionError("Native prediction is nonfinite")
            expected = 0 if prior is None else expected_vjps
            if len(traces) != expected or any(item["padded_residual_max"] != 0 for item in traces):
                raise AssertionError("Missing native guidance steps or nonzero padded-channel residual")
            values = normalized.cpu().numpy()[0]
            row.update(finite=True, normalized_sha256=fingerprint(values), decoded_sha256=fingerprint(decoded))
            return values, decoded

        run("discarded_warmup", None)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        baseline = run("no_prefix_1", None)
        repeated = run("no_prefix_2", None)
        zero = run("zero_mask_guided", {"target": target, "mask": np.zeros_like(mask), "prefix_len": PREFIX})
        guided = run("real_mask_guided_1", {"target": target, "mask": mask, "prefix_len": PREFIX})
        guided_repeat = run("real_mask_guided_2", {"target": target, "mask": mask, "prefix_len": PREFIX})
        comparisons = {"baseline_repeat": (baseline, repeated), "zero_mask_invariance": (baseline, zero),
                       "guided_repeat": (guided, guided_repeat), "guidance_effect": (zero, guided),
                       "total_mode_effect": (baseline, guided)}
        evidence["comparisons"] = {name: {
            "normalized_all_channels": difference(left[0], right[0]),
            "normalized_real_regions": region_differences(left[0][:, :16], right[0][:, :16]),
            "absolute_command_regions": region_differences(left[1], right[1])}
            for name, (left, right) in comparisons.items()}
        for name in ("baseline_repeat", "zero_mask_invariance", "guided_repeat"):
            left, right = comparisons[name]
            if not np.allclose(left[0], right[0], atol=ATOL, rtol=RTOL):
                raise AssertionError(f"{name} exceeded the fixed normalized-action tolerance")
        if any(item["residual_max"] != 0 or item["vjp_max"] != 0
               for item in evidence["checks"][3]["vjp_trace"]):
            raise AssertionError("Zero-mask native guidance had a nonzero residual or VJP")
        real_traces = evidence["checks"][4]["vjp_trace"]
        if not any(item["residual_max"] > 0 and item["vjp_max"] > 0 for item in real_traces):
            raise AssertionError("Real-channel guidance did not exercise a nonzero VJP")
        # Compare two grad-enabled paths so any baseline/grad-kernel rounding
        # difference is not mislabeled as a nonzero guidance effect.
        effect = difference(zero[0][:, :16], guided[0][:, :16])["max_absolute"]
        floor = max(difference(baseline[0][:, :16], repeated[0][:, :16])["max_absolute"],
                    difference(guided[0][:, :16], guided_repeat[0][:, :16])["max_absolute"])
        evidence.update(guidance_effect_max=effect, repeatability_floor_max=floor,
                        guidance_effect_pair=["zero_mask_guided", "real_mask_guided_1"],
                        effect_above_repeatability_floor=effect > floor)
        if effect <= floor:
            raise AssertionError("Guidance effect was not distinguishable from measured repeat noise")
        evidence["prefix_target_error"] = {
            "baseline": difference(baseline[1][:PREFIX], absolute_prior[:PREFIX]),
            "guided": difference(guided[1][:PREFIX], absolute_prior[:PREFIX])}
        evidence["complete"] = True
    except Exception as exc:
        error = exc
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        if policy is not None:
            policy.reset()
        evidence["elapsed_s"] = time.monotonic() - started
        write_json(args.output, evidence)
    if error is not None:
        raise error
    print(json.dumps({"complete": True, "scope": evidence["scope"], "output": str(args.output),
                      "inference_calls": len(evidence["checks"])}))


if __name__ == "__main__":
    main()
