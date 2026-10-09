"""Current-frame OpenArm adapter for the Pillow RLDX-1 acceptance checkpoint.

The native policy owns preprocessing, sampling and absolute-action decoding.
OpenArm's shared caller owns execution windows, pacing and chunk identity.
"""

from pathlib import Path
import time

import numpy as np

from openarm_policy_runtime import Backend, Observation, Prediction


CAMERAS = ("wrist_right", "wrist_left", "ceiling", "head_left")
GROUPS = ("right_arm", "right_gripper", "left_arm", "left_gripper")
DIMS = (7, 1, 7, 1)
HORIZON = 32
INTERVAL_NS = 33_333_333  # 30 Hz targets, independent of inference request rate.
RTC_REASON = (
    "Only none and experimental guided RTC are supported. Trained RTC requires "
    "matching rtc_training_max_delay training and numerical validation; executor "
    "is a feedback source, not a native RLDX sampling mode."
)


def require_supported_rtc(mode):
    if mode not in ("none", "guided"):
        raise ValueError(RTC_REASON)


def validate_manifest(manifest):
    expected = {
        "format_version": 1, "horizon": HORIZON, "action_hz": 30,
        "video_keys": list(CAMERAS), "state_action_keys": list(GROUPS),
        "group_dims": list(DIMS), "absolute_command": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"OpenArm checkpoint metadata {key}: expected {value!r}")


def _value(value):
    return getattr(value, "value", value)


def validate_policy(policy, rtc=None):
    """Fail closed on a different embodiment, history or action representation."""
    tag = policy.embodiment_tag
    if getattr(tag, "name", None) != "GENERAL_EMBODIMENT":
        raise ValueError("This checkpoint must use GENERAL_EMBODIMENT")
    config = policy.model.config
    if config.action_horizon != HORIZON or config.video_length != 1:
        raise ValueError("Expected action_horizon=32 and video_length=1")
    if any(getattr(config, key, False) for key in ("use_memory", "use_motion", "use_physics")):
        raise ValueError("Memory, motion and physics addons are unsupported")
    if getattr(config, "physics_keys", []) or policy.use_memory:
        raise ValueError("Checkpoint requires unsupported auxiliary inputs")
    require_supported_rtc(policy.rtc_inference_mode)
    if policy.rtc_inference_mode != ("guided" if rtc is not None else "none"):
        raise ValueError("Guided mode requires the explicit executor RTC adapter")
    modalities = policy.modality_configs
    for name, keys, offsets in (
        ("video", CAMERAS, [0]), ("state", GROUPS, [0]),
        ("action", GROUPS, list(range(HORIZON))),
    ):
        item = modalities[name]
        if tuple(item.modality_keys) != keys or list(item.delta_indices) != offsets:
            raise ValueError(f"Checkpoint {name} order/history differs from OpenArm contract")
    language = modalities["language"]
    if list(language.delta_indices) != [0] or language.modality_keys != [policy.language_key]:
        raise ValueError("Expected one current language instruction")
    sap = policy.processor.state_action_processor
    if not sap.use_relative_action:
        raise ValueError("Checkpoint processor must decode relative arms to absolute targets")
    stats = sap.norm_params[tag.value]
    for kind in ("state", "action"):
        for key, width in zip(GROUPS, DIMS):
            if int(stats[kind][key]["dim"]) != width:
                raise ValueError(f"Checkpoint {kind}.{key} statistics have the wrong dimension")
    action_configs = modalities["action"].action_configs
    if action_configs is None or len(action_configs) != len(GROUPS):
        raise ValueError("Checkpoint must declare all four action representations")
    for index, (key, spec) in enumerate(zip(GROUPS, action_configs)):
        rep = "relative" if index in (0, 2) else "absolute"
        if (_value(spec.rep) != rep or _value(spec.type) != "non_eef"
                or _value(spec.format) != "default" or (spec.state_key or key) != key):
            raise ValueError(f"Checkpoint action representation for {key} differs from training")


class OpenArmBackend(Backend):
    name = "rldx1-openarm"
    camera_fields = tuple("camera_" + key for key in CAMERAS)

    def __init__(self, policy, manifest, *, warmup_sample=None, warmup_steps=0, prompt="", rtc=None):
        validate_manifest(manifest)
        validate_policy(policy, rtc)
        if warmup_steps < 0 or (warmup_steps and warmup_sample is None):
            raise ValueError("Positive warmup_steps requires a recorded warmup_sample")
        self.policy = policy
        self.warmup_sample = Path(warmup_sample) if warmup_sample is not None else None
        self.warmup_steps = warmup_steps
        self.prompt = prompt
        self.rtc = rtc

    def warmup(self):
        if not self.warmup_steps:
            return
        try:
            with np.load(self.warmup_sample, allow_pickle=False) as sample:
                prompt = str(sample["prompt"].item()) if "prompt" in sample else self.prompt
                observation = Observation(
                    timestamp=time.time_ns(), qpos=sample["qpos"].copy()[None],
                    cameras={key: sample[key].copy()[None] for key in self.camera_fields},
                    prompt=prompt, delta_indices=(0,),
                )
            for _ in range(self.warmup_steps):
                self.predict(observation)  # Discard; never send synthetic actions to an executor.
        finally:
            self.reset("warmup")

    def predict(self, observation):
        if self.policy is None:
            raise RuntimeError("Backend is closed")
        started_ns = time.time_ns()
        index = observation.current_index()
        if observation.qpos.ndim != 2 or observation.qpos.shape[1] != 16:
            raise ValueError("OpenArm qpos must have shape [history,16]")
        current = np.asarray(observation.qpos[index], dtype=np.float32)
        if not np.isfinite(current).all():
            raise ValueError("OpenArm state must be finite")
        video = {}
        for key, field in zip(CAMERAS, self.camera_fields):
            frames = observation.cameras[field]
            if (frames.ndim != 4 or frames.shape[0] != observation.qpos.shape[0]
                    or frames.shape[-1] != 3 or min(frames.shape[1:3]) <= 0
                    or frames.dtype != np.uint8):
                raise ValueError(f"{field} must be RGB uint8 [history,height,width,3]")
            video[key] = frames[index].copy()[None, None]
        state, start = {}, 0
        for key, width in zip(GROUPS, DIMS):
            state[key] = current[start:start + width].copy()[None, None]
            start += width
        native = {
            "video": video, "state": state,
            "language": {self.policy.language_key: [[observation.prompt]]},
        }
        options = {"session_ids": ["openarm"]}
        execution, rtc_log = None, {"mode": "none", "used": False}
        if self.rtc is not None:
            prepared = self.rtc.prepare(observation, started_ns)
            if prepared.reset_native:
                self.policy.reset()
            if prepared.skip:
                return Prediction(None, interval_ns=INTERVAL_NS, log_data={"rtc": prepared.log_data})
            options.update(prepared.options)
            execution, rtc_log = prepared.execution, prepared.log_data
        started = time.perf_counter()
        actions, _ = self.policy.get_action(native, options=options)
        parts = []
        for key, width in zip(GROUPS, DIMS):
            part = np.asarray(actions[key], dtype=np.float32)
            if part.shape != (1, HORIZON, width) or not np.isfinite(part).all():
                raise ValueError(f"Native decoded {key} must be finite [1,32,{width}]")
            parts.append(part[0])
        # Native decode_action already denormalizes and adds raw state to relative arms.
        positions = np.concatenate(parts, axis=-1)
        return Prediction(
            positions=positions, interval_ns=INTERVAL_NS,
            execution=execution,
            timing={"rldx_predict_ms": (time.perf_counter() - started) * 1000},
            log_data={"rtc": rtc_log, "action_hz": 30, "action_horizon": HORIZON},
        )

    def reset(self, reason):
        if self.policy is not None:
            self.policy.reset()
        if self.rtc is not None:
            self.rtc.reset()
        return False

    def close(self):
        self.reset("close")
        self.policy = None
