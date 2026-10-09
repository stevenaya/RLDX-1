"""Executor-aligned native Jacobian guidance, with no hard-prefix guarantee.

The caller owns windows/ACKs. This adapter only snapshots absolute adopted plans,
aligns their time grid and encodes them through the checkpoint's native processor.
Training/state normalization and native output-decoder clipping are unchanged.
"""

from copy import copy
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


@dataclass
class RTCRequest:
    options: dict = field(default_factory=lambda: {"executor_rtc": None})
    execution: dict | None = None
    log_data: dict = field(default_factory=dict)
    skip: bool = False
    reset_native: bool = False


class ExecutionRTC:
    """One accepted plan + predecessor; synchronous single-session Backend use.

    Guidance weight is 1 on [0,F), native exponential taper on [F,16), and
    zero on [16,32). Even weight 1 is soft attraction, not hard inpainting.
    Delays F>16 are unsupported by native d<=s<=H-d and are skipped explicitly.
    A finished adopted plan holds its last command, matching the executor.
    """

    def __init__(self, policy, *, margin_ms=10, max_lateness_ms=50):
        if (not math.isfinite(margin_ms) or margin_ms < 0
                or not math.isfinite(max_lateness_ms) or max_lateness_ms < 0):
            raise ValueError("RTC margins must be finite and nonnegative")
        if (policy.rtc_inference_mode != "guided" or policy.rtc_exec_horizon != 16
                or policy.rtc_inference_delay != 1):
            raise ValueError("Executor RTC requires native guided mode, delay=1, exec_horizon=16")
        self.codec = PriorCodec(policy)
        self.margin_ns = int(margin_ms * 1e6)
        self.max_lateness_ns = int(max_lateness_ms * 1e6)
        self.delay_ns = 0.0
        self.reset()

    def reset(self):
        self.plan = self.previous_plan = None
        self.context = None
        self.last_sample = None
        self.latest_receipt = -1
        self.boundary_ns = None
        self.delay_ns = 0.0

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

    def prepare(self, observation, started_ns):
        meta = observation.metadata
        context = (observation.prompt, meta.get("inference_trial_id"), meta.get("episode_attempt_id"))
        changed = self.context != context or bool(meta.get("execution_restart"))
        if changed:
            self.reset()
            self.context = context
            self.boundary_ns = int(started_ns)
        origin = int(observation.timestamp)
        if origin <= 0 or started_ns < origin:
            raise ValueError("RTC requires positive observation timestamps not later than inference start")
        result = RTCRequest(reset_native=changed)
        result.log_data = dict(source="executor", mode="guided", hard_prefix=False,
                               used=False, reason="bootstrap", delay_ms=self.delay_ns / 1e6)
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
        if raw_plan.get("chunk_id") == "" and len(raw_plan.get("positions", [])) == 0:
            self.plan = self.previous_plan = None
            result.log_data["reason"] = "no_adopted_plan"
            return result
        plan = self.snapshot(raw_plan)
        sample_start = int(plan.get("inference_started_timestamp_ns", 0))
        received = int(plan.get("received_timestamp_ns", 0))
        sample_id = plan.get("sample_chunk_id")
        plan_attempt = plan.get("episode_attempt_id")
        reason = None
        if not isinstance(sample_id, str) or not sample_id:
            reason = "missing_sample_identity"
        elif plan_attempt is not None and plan_attempt != context[2]:
            reason = "attempt_mismatch"
        elif sample_start < self.boundary_ns or received < sample_start or received > started_ns:
            reason = "stale_or_invalid_plan_timing"
        elif received < self.latest_receipt:
            reason = "superseded_plan"
        if reason:
            result.skip, result.execution = True, None
            result.log_data["reason"] = reason
            return result
        self.latest_receipt = received
        if sample_id != self.last_sample:
            elapsed = received - sample_start
            self.delay_ns = .9 * self.delay_ns + .1 * elapsed if self.delay_ns else float(elapsed)
            self.last_sample = sample_id
        result.log_data["delay_ms"] = self.delay_ns / 1e6
        prefix = max(1, math.ceil((started_ns - origin + self.delay_ns + self.margin_ns) / INTERVAL_NS),
                     int(meta.get("rtc_frozen_steps", 0)))
        if prefix > GUIDANCE_END:
            result.skip, result.execution = True, None
            result.log_data.update(reason="delay_exceeds_native_guided_limit", prefix_steps=prefix)
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
        result.options = {"executor_rtc": {"target": target, "mask": mask, "prefix_len": prefix}}
        result.execution.update(action_window_start=prefix, action_origin_timestamp_ns=origin,
                                takeover_timestamp_ns=origin + prefix * INTERVAL_NS,
                                based_on_chunk_id=plan["chunk_id"])
        held = timestamps > plan["start_timestamp_ns"] + (len(plan["positions"]) - 1) * plan["interval_ns"]
        result.log_data.update(used=True, reason="guided", prefix_steps=prefix,
                               ramp_steps=GUIDANCE_END-prefix, guidance_end=GUIDANCE_END,
                               held_endpoint_steps=int(held.sum()), delay_ms=self.delay_ns / 1e6)
        return result
