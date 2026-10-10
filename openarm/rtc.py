"""Executor-aligned guided and gradient-free prior conditioning.

The caller owns windows/ACKs. This adapter only snapshots absolute adopted plans,
aligns their time grid and encodes them through the checkpoint's native processor.
Training/state normalization and native output-decoder clipping are unchanged.
"""

from copy import copy
from collections import OrderedDict
from dataclasses import dataclass, field
import math

import numpy as np


GROUPS = ("right_arm", "right_gripper", "left_arm", "left_gripper")
SLICES = (slice(0, 7), slice(7, 8), slice(8, 15), slice(15, 16))
HORIZON = 32
INTERVAL_NS = 33_333_333
GUIDANCE_END = 16  # Native s=16: taper ends here; [16,32) is freely generated.


class PriorCodec:
    """Scoped unclipped prior encoder; native output decoding is NOT replaced."""

    def __init__(self, policy):
        self.tag = policy.embodiment_tag.value
        self.native_sap = policy.processor.state_action_processor
        self.sap = copy(policy.processor.state_action_processor)
        self.sap.clip_outliers = False
        self.dim = int(policy.model.action_model.action_dim)
        if self.dim < 16:
            raise ValueError("OpenArm guidance needs at least 16 model action channels")
        config = self.sap.modality_configs[self.tag]["action"]
        if tuple(config.modality_keys) != GROUPS:
            raise ValueError("RTC requires the verified OpenArm action group order")
        if config.mean_std_embedding_keys:
            raise ValueError("This prior inverse audit supports OpenArm min-max statistics only")

    def encode(self, absolute, current):
        absolute = np.asarray(absolute, dtype=np.float32)
        current = np.asarray(current, dtype=np.float32)
        if (absolute.shape != (HORIZON, 16) or current.shape != (16,)
                or not np.isfinite(absolute).all() or not np.isfinite(current).all()):
            raise ValueError("RTC prior/state must be finite [32,16] / [16]")
        actions = {key: absolute[:, section] for key, section in zip(GROUPS, SLICES)}
        state = {key: current[section][None] for key, section in zip(GROUPS, SLICES)}
        encoded = self.sap.apply_action(actions, self.tag, state=state)
        target = np.zeros((1, HORIZON, self.dim), dtype=np.float32)
        for key, section in zip(GROUPS, SLICES):
            target[0, :, section] = encoded[key]
        if not np.isfinite(target).all():
            raise ValueError("Checkpoint produced nonfinite RTC normalization")
        # A collapsed checkpoint range has no inverse. Reject a lossy prior;
        # a nominally valid min=max channel is accepted only at its constant.
        decoded = self.decode_unclipped(target, current)
        if not np.allclose(decoded, absolute, rtol=2e-5, atol=2e-5):
            raise ValueError("RTC prior is not invertible through checkpoint normalization")
        mask = np.zeros_like(target, dtype=bool)
        mask[..., :16] = True
        return target, mask

    def decode_unclipped(self, normalized, current):
        """Audit inverse only: exposes encode loss without native output clipping.

        The actual native decoder still clips min-max outputs to [-1,1]. This
        roundtrip proves the conditioning target, never a frozen output prefix.
        """
        normalized = np.asarray(normalized, dtype=np.float32)[0]
        current = np.asarray(current, dtype=np.float32)
        result = np.empty((HORIZON, 16), dtype=np.float32)
        for index, (key, section) in enumerate(zip(GROUPS, SLICES)):
            params = self.sap.norm_params[self.tag]["action"][key]
            lower, upper = np.asarray(params["min"]), np.asarray(params["max"])
            result[:, section] = (normalized[:, section] + 1) * (upper - lower) / 2 + lower
            if index in (0, 2):
                result[:, section] += current[section]
        return result

    def validate_inpaint(self, target, current, prefix_len, ramp_steps=0):
        """Reject frozen values that native output clipping cannot preserve.

        The first linear-ramp point also has zero update strength. This FP32
        preflight is not an exact decoded BF16 equality guarantee: quantization
        remains and is measured separately, without changing native decoding.
        """
        frozen = int(prefix_len) + bool(ramp_steps)
        if not 1 <= frozen <= HORIZON:
            raise ValueError("Invalid inpaint frozen region")
        real = np.asarray(target, dtype=np.float32)[0, :frozen, :16]
        if not np.isfinite(real).all() or np.any(np.abs(real) > 1):
            raise ValueError("Inpaint frozen prior is outside native decoder [-1,1] bounds")
        groups = {key: np.asarray(target)[0, :, section] for key, section in zip(GROUPS, SLICES)}
        state = {key: np.asarray(current)[section][None] for key, section in zip(GROUPS, SLICES)}
        decoded = self.native_sap.unapply_action(groups, self.tag, state=state)
        decoded = np.concatenate([decoded[key] for key in GROUPS], axis=-1)
        expected = self.decode_unclipped(target, current)
        error = float(np.abs(decoded[:frozen] - expected[:frozen]).max())
        if not np.allclose(decoded[:frozen], expected[:frozen], rtol=2e-5, atol=2e-5):
            raise ValueError("Inpaint frozen prior is not preserved by native output decoding")
        return {"frozen_decode_fp32_error": error,
                "decoded_prefix_scope": "native BF16 quantization; no output replacement"}


@dataclass
class RTCRequest:
    options: dict = field(default_factory=lambda: {"executor_rtc": None})
    execution: dict | None = None
    log_data: dict = field(default_factory=dict)
    skip: bool = False
    reset_native: bool = False


class ExecutionRTC:
    """Bounded episode state and separate effective-path latency calibration."""

    MAX_SAMPLES = 32
    MAX_CALIBRATIONS = 8

    def __init__(self, policy, *, mode="guided", guidance_steps=None,
                 inpaint_ramp_steps=0, margin_ms=10, max_lateness_ms=50):
        if (not math.isfinite(margin_ms) or margin_ms < 0
                or not math.isfinite(max_lateness_ms) or max_lateness_ms < 0):
            raise ValueError("RTC margins must be finite and nonnegative")
        if (policy.rtc_inference_mode != "guided" or policy.rtc_exec_horizon != 16
                or policy.rtc_inference_delay != 1):
            raise ValueError("Executor RTC requires native guided mode, delay=1, exec_horizon=16")
        if mode not in ("guided", "inpaint"):
            raise ValueError("Executor RTC supports guided or inpaint")
        head = policy.model.action_model
        custom = getattr(head, "denoising_timesteps", None)
        self.denoising_steps = len(custom) if custom is not None else int(head.num_inference_timesteps)
        default_guidance = head._rtc.jacobian_steps_only
        effective = default_guidance if guidance_steps is None else guidance_steps
        effective = self.denoising_steps if effective is None else effective
        if isinstance(effective, bool) or not isinstance(effective, int) or not 1 <= effective <= self.denoising_steps:
            raise ValueError("guidance_steps must be within the actual denoising step count")
        if (isinstance(inpaint_ramp_steps, bool) or not isinstance(inpaint_ramp_steps, int)
                or not 0 <= inpaint_ramp_steps < HORIZON):
            raise ValueError("inpaint_ramp_steps must be an integer in [0,31]")
        if mode == "guided" and inpaint_ramp_steps:
            raise ValueError("inpaint_ramp_steps applies only to inpaint")
        self.mode, self.guidance_steps = mode, effective
        self.inpaint_ramp_steps = inpaint_ramp_steps
        self.backend_key = (str(policy.model.device),
                            str(getattr(policy.model.config, "_attn_implementation", "sdpa")),
                            tuple(custom) if custom is not None else None,
                            float(getattr(head._rtc, "jacobian_beta", 5.0)))
        self.codec = PriorCodec(policy)
        self.margin_ns = int(margin_ms * 1e6)
        self.max_lateness_ns = int(max_lateness_ms * 1e6)
        self.latencies_ns = OrderedDict()
        self.active_key = None
        self.reset()

    @property
    def delay_ns(self):
        return self.latencies_ns.get(self.active_key, 0.0)

    def reset(self, *, clear_calibration=False):
        self.plan = self.previous_plan = None
        self.context = None
        self.latest_receipt = -1
        self.boundary_ns = None
        self.pending = OrderedDict()
        self.acknowledged = OrderedDict()
        if clear_calibration:
            self.latencies_ns.clear()

    def path_key(self, observation, mode):
        shapes = tuple((key, tuple(value.shape[1:])) for key, value in sorted(observation.cameras.items()))
        settings = (mode, self.denoising_steps,
                    self.guidance_steps if mode == "guided" else 0,
                    self.inpaint_ramp_steps if mode == "inpaint" else 0,
                    self.backend_key, shapes)
        return repr(settings)

    @staticmethod
    def _put_bounded(mapping, key, value, limit):
        mapping[key] = value
        mapping.move_to_end(key)
        while len(mapping) > limit:
            mapping.popitem(last=False)

    def _calibrate(self, key, elapsed, *, replace=False):
        old = self.latencies_ns.get(key)
        value = float(elapsed) if old is None or replace else .9 * old + .1 * elapsed
        self._put_bounded(self.latencies_ns, key, value, self.MAX_CALIBRATIONS)

    def latency_snapshot(self):
        return {key: value / 1e6 for key, value in self.latencies_ns.items()}

    def seed_calibration(self, key, samples_ns):
        if not samples_ns or any(not math.isfinite(x) or x <= 0 for x in samples_ns):
            raise ValueError("Warmup calibration requires finite positive measured durations")
        self._calibrate(key, float(np.median(samples_ns)), replace=True)

    def record_prediction(self, observation, started_ns, elapsed_ns, trace, expected_mode):
        """Associate actual execution with caller identity; no guessed feedback mode."""
        expected_vjp = self.guidance_steps if expected_mode == "guided" else 0
        if (trace.get("effective_mode") != expected_mode or trace.get("vjp_calls") != expected_vjp
                or trace.get("denoising_steps") != self.denoising_steps):
            raise ValueError("Native RTC effective path/VJP count differs from requested execution")
        key = self.path_key(observation, expected_mode)
        sample_id = observation.metadata.get("chunk_id")
        if isinstance(sample_id, str) and sample_id:
            if sample_id in self.pending or sample_id in self.acknowledged:
                raise ValueError("Caller chunk_id was reused within an RTC episode")
            self._put_bounded(self.pending, sample_id, (int(started_ns), key), self.MAX_SAMPLES)
        if not math.isfinite(elapsed_ns) or elapsed_ns <= 0:
            raise ValueError("Local compute timing requires a positive monotonic duration")
        return dict(trace, calibration_key=key, elapsed_ms=elapsed_ns/1e6)

    def _acknowledge(self, plan):
        sample_id = plan.get("sample_chunk_id")
        if not isinstance(sample_id, str) or not sample_id:
            return "missing_sample_identity"
        start, received = int(plan.get("inference_started_timestamp_ns", 0)), int(plan.get("received_timestamp_ns", 0))
        entry = self.pending.get(sample_id) or self.acknowledged.get(sample_id)
        if entry is None:
            return "untracked_sample_identity"
        if entry[0] != start:
            return "sample_start_mismatch"
        if sample_id in self.acknowledged and entry[2] != received:
            return "sample_receipt_mismatch"
        if sample_id in self.pending:
            self._calibrate(entry[1], received-start)
            self.pending.pop(sample_id)
            self._put_bounded(self.acknowledged, sample_id, (*entry, received), self.MAX_SAMPLES)
        return None

    @staticmethod
    def snapshot(plan):
        points = np.array(plan["positions"], dtype=np.float32, copy=True)
        start, interval = int(plan["start_timestamp_ns"]), int(plan["interval_ns"])
        chunk = plan.get("chunk_id")
        if (not isinstance(chunk, str) or not chunk or start <= 0 or interval <= 0
                or points.ndim != 2 or points.shape[1] != 16 or len(points) == 0
                or not np.isfinite(points).all()):
            raise ValueError("RTC plan needs chunk_id, positive time grid and finite [N,16] targets")
        result = dict(plan)
        result.update(positions=points, start_timestamp_ns=start, interval_ns=interval)
        return result

    @staticmethod
    def sample(plan, timestamps):
        points = plan["positions"]
        relative = np.asarray(timestamps, dtype=np.int64) - plan["start_timestamp_ns"]
        if (relative < 0).any():
            raise ValueError("Cannot sample before the adopted plan starts")
        source = np.arange(len(points), dtype=np.int64) * plan["interval_ns"]
        return np.stack([np.interp(relative, source, column) for column in points.T], -1).astype(np.float32)

    def prepare(self, observation, started_ns, *, calibration_only=False):
        meta = observation.metadata
        context = (observation.prompt, meta.get("inference_trial_id"), meta.get("episode_attempt_id"))
        changed = self.context != context or bool(meta.get("execution_restart"))
        if changed:
            self.reset()
            self.context = context
            self.boundary_ns = int(started_ns)
        self.active_key = self.path_key(observation, self.mode)
        origin = int(observation.timestamp)
        if origin <= 0 or started_ns < origin:
            raise ValueError("RTC requires positive observation timestamps not later than inference start")
        result = RTCRequest(reset_native=changed)
        result.log_data = dict(source="executor", mode=self.mode, hard_prefix=self.mode == "inpaint",
                               used=False, reason="bootstrap",
                               delay_ms=self.delay_ns / 1e6 if self.active_key in self.latencies_ns else None,
                               calibration_key=self.active_key, calibrated=self.active_key in self.latencies_ns,
                               calibration_only=bool(calibration_only))
        result.execution = dict(action_window_start=0, action_origin_timestamp_ns=0,
                                takeover_timestamp_ns=0, based_on_chunk_id="",
                                inference_started_timestamp_ns=int(started_ns),
                                max_lateness_ns=self.max_lateness_ns)
        if changed or observation.execution_plan is None:
            return result
        # Pinned eval dora_openarm_actions_executor/main.py:feedback emits an
        # empty chunk/positions acknowledgement when no plan has been adopted.
        # It withdraws any stored coverage; never reuse it as a predecessor.
        raw_plan = observation.execution_plan
        withdrawn = raw_plan.get("chunk_id") == "" and len(raw_plan.get("positions", [])) == 0
        if withdrawn:
            self.plan = self.previous_plan = None
            result.log_data["reason"] = "no_adopted_plan"
            if not raw_plan.get("sample_chunk_id"):
                return result
        plan = raw_plan if withdrawn else self.snapshot(raw_plan)
        sample_start = int(plan.get("inference_started_timestamp_ns", 0))
        received = int(plan.get("received_timestamp_ns", 0))
        sample_id = plan.get("sample_chunk_id")
        plan_attempt = plan.get("episode_attempt_id")
        reason = None
        if not isinstance(sample_id, str) or not sample_id:
            reason = "missing_sample_identity"
        elif plan_attempt is not None and plan_attempt != context[2]:
            reason = "attempt_mismatch"
        elif sample_start < self.boundary_ns or received <= sample_start or received > started_ns:
            reason = "stale_or_invalid_plan_timing"
        elif received < self.latest_receipt:
            reason = "superseded_plan"
        if reason is None:
            reason = self._acknowledge(plan)
        if reason:
            result.skip, result.execution = True, None
            result.log_data["reason"] = reason
            return result
        self.latest_receipt = received
        calibrated = self.active_key in self.latencies_ns
        result.log_data.update(calibrated=calibrated,
                               delay_ms=self.delay_ns / 1e6 if calibrated else None)
        if withdrawn:
            return result
        # Only discarded local warmup may exercise an as-yet unmeasured active
        # path. Live requests cannot schedule a prefix using unknown latency.
        if not calibrated and not calibration_only:
            result.skip, result.execution = True, None
            result.log_data["reason"] = "latency_uncalibrated"
            return result
        prefix = max(1, math.ceil((started_ns - origin + self.delay_ns + self.margin_ns) / INTERVAL_NS),
                     int(meta.get("rtc_frozen_steps", 0)))
        limit = GUIDANCE_END if self.mode == "guided" else min(HORIZON-1, HORIZON-self.inpaint_ramp_steps)
        if prefix > limit:
            result.skip, result.execution = True, None
            result.log_data.update(reason=f"delay_exceeds_native_{self.mode}_limit", prefix_steps=prefix)
            return result
        if self.plan is None or self.plan["chunk_id"] != plan["chunk_id"]:
            self.previous_plan, self.plan = self.plan, plan
        else:
            self.plan = plan
        timestamps = origin + np.arange(HORIZON, dtype=np.int64) * INTERVAL_NS
        before = timestamps < plan["start_timestamp_ns"]
        targets = np.empty((HORIZON, 16), dtype=np.float32)
        if before.any():
            old = self.previous_plan
            if old is None or timestamps[0] < old["start_timestamp_ns"]:
                result.skip, result.execution = True, None
                result.log_data["reason"] = "missing_pre_handoff_coverage"
                return result
            targets[before] = self.sample(old, timestamps[before])
        targets[~before] = self.sample(plan, timestamps[~before])
        current = observation.qpos[observation.current_index()]
        target, mask = self.codec.encode(targets, current)
        if self.mode == "inpaint":
            try:
                result.log_data.update(self.codec.validate_inpaint(target, current, prefix, self.inpaint_ramp_steps))
            except ValueError as error:
                result.skip, result.execution = True, None
                result.log_data.update(reason="inpaint_prior_not_preservable", detail=str(error))
                return result
        result.options = {"executor_rtc": {"target": target, "mask": mask, "prefix_len": prefix,
                           "mode": self.mode, "guidance_steps": self.guidance_steps if self.mode == "guided" else None,
                           "ramp_steps": self.inpaint_ramp_steps}}
        result.execution.update(action_window_start=prefix, action_origin_timestamp_ns=origin,
                                takeover_timestamp_ns=origin + prefix * INTERVAL_NS,
                                based_on_chunk_id=plan["chunk_id"])
        held = timestamps > plan["start_timestamp_ns"] + (len(plan["positions"]) - 1) * plan["interval_ns"]
        result.log_data.update(used=True, reason=self.mode, prefix_steps=prefix,
                               ramp_steps=GUIDANCE_END-prefix if self.mode == "guided" else self.inpaint_ramp_steps,
                               guidance_end=GUIDANCE_END if self.mode == "guided" else prefix+self.inpaint_ramp_steps,
                               held_endpoint_steps=int(held.sum()),
                               delay_ms=self.delay_ns / 1e6 if calibrated else None)
        return result
