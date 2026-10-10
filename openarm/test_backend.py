"""CPU contract tests; no Torch import, model weights or robot execution.

Run inside the isolated model job with the pinned eval runtime on PYTHONPATH:
    python -m unittest openarm.test_backend -v
"""

from copy import deepcopy
from enum import Enum
import json
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from openarm_policy_runtime import ModelSession, Observation, serve_connection

from openarm.backend import CAMERAS, DIMS, GROUPS, OpenArmBackend, require_supported_rtc
from openarm.serve import parse_args


class Tag(Enum):
    GENERAL_EMBODIMENT = "general_embodiment"


def manifest():
    return dict(format_version=1, horizon=32, action_hz=30, video_keys=list(CAMERAS),
                state_action_keys=list(GROUPS), group_dims=list(DIMS), absolute_command=True)


class FakePolicy:
    """Native-policy stand-in returns already decoded absolute trajectories."""

    def __init__(self):
        self.embodiment_tag = Tag.GENERAL_EMBODIMENT
        self.model = NS(config=NS(action_horizon=32, video_length=1), device="cpu",
                        action_model=NS(action_dim=64, num_inference_timesteps=4,
                                        _rtc=NS(jacobian_steps_only=3)))
        self.use_memory = False
        self.rtc_inference_mode = "none"
        self.rtc_inference_delay, self.rtc_exec_horizon = 1, 16
        self.language_key = "annotation.human.action.task_description"
        self.modality_configs = {
            "video": NS(modality_keys=list(CAMERAS), delta_indices=[0]),
            "state": NS(modality_keys=list(GROUPS), delta_indices=[0]),
            "action": NS(modality_keys=list(GROUPS), delta_indices=list(range(32)),
                         mean_std_embedding_keys=[],
                         action_configs=[NS(rep="relative" if i in (0, 2) else "absolute",
                                            type="non_eef", format="default", state_key=key)
                                         for i, key in enumerate(GROUPS)]),
            "language": NS(modality_keys=[self.language_key], delta_indices=[0]),
        }
        norms = {kind: {key: {"dim": width, "min": -np.ones(width), "max": np.ones(width)}
                        for key, width in zip(GROUPS, DIMS)}
                 for kind in ("state", "action")}
        def encode(actions, tag, state):
            return {key: value-state[key] if key.endswith("arm") else value.copy()
                    for key, value in actions.items()}
        def decode(actions, tag, state):
            return {key: np.clip(value, -1, 1)+state[key] if key.endswith("arm") else np.clip(value, -1, 1)
                    for key, value in actions.items()}
        self.processor = NS(state_action_processor=NS(
            use_relative_action=True, clip_outliers=True, norm_params={"general_embodiment": norms},
            modality_configs={"general_embodiment": self.modality_configs}, apply_action=encode, unapply_action=decode))
        self.calls, self.reset_count = [], 0
        self.bad_output = False

    def get_action(self, observation, options):
        self.calls.append((deepcopy(observation), options))
        actions = {key: np.repeat(value, 32, axis=1) + np.arange(32)[None, :, None] / 100
                   for key, value in observation["state"].items()}
        if self.bad_output:
            actions["right_arm"][0, 2, 1] = np.nan
        prior = options.get("executor_rtc")
        mode = prior.get("mode", "guided") if prior is not None else "none"
        steps = (prior.get("guidance_steps") or 3) if mode == "guided" else 0
        return actions, {"rtc": dict(effective_mode=mode, vjp_calls=steps, guidance_steps=steps,
                                      denoising_steps=4, prefix_len=prior["prefix_len"] if prior else 0,
                                      ramp_steps=prior.get("ramp_steps", 0) if prior else 0,
                                      timesteps=[0, .25, .5, .75, 1])}

    def reset(self):
        self.reset_count += 1


def observation():
    return Observation(
        123, np.stack([np.arange(16), np.arange(16) + 100]).astype(np.float32),
        {"camera_" + key: np.stack([np.full((2, 3, 3), i, dtype=np.uint8),
                                    np.full((2, 3, 3), i + 20, dtype=np.uint8)])
         for i, key in enumerate(CAMERAS)}, "pillow", delta_indices=(0, -1),
    )


class BackendTests(unittest.TestCase):
    def test_current_index_order_and_full_decoded_output(self):
        policy = FakePolicy()
        backend = OpenArmBackend(policy, manifest())
        obs = observation()
        obs.metadata.update(action_window_start=7, action_window_size=2)
        result = backend.predict(obs)
        self.assertEqual(result.positions.shape, (32, 16))
        self.assertEqual(result.interval_ns, 33_333_333)
        self.assertIsNone(result.execution)
        np.testing.assert_allclose(result.positions[0], np.arange(16))
        np.testing.assert_allclose(result.positions[-1], np.arange(16) + .31)
        native, options = policy.calls[0]
        self.assertEqual(list(native["video"]), list(CAMERAS))
        self.assertEqual(list(native["state"]), list(GROUPS))
        self.assertEqual(native["language"][policy.language_key], [["pillow"]])
        for index, key in enumerate(CAMERAS):
            np.testing.assert_array_equal(native["video"][key], index)
        self.assertNotIn("action_prefix", options)
        self.assertTrue(all(value.shape == (1, 1, dim)
                            for value, dim in zip(native["state"].values(), DIMS)))

    def test_invalid_inputs_and_outputs_are_rejected(self):
        policy = FakePolicy()
        backend = OpenArmBackend(policy, manifest())
        obs = observation()
        obs.delta_indices = (-2, -1)
        with self.assertRaisesRegex(ValueError, "offset 0"):
            backend.predict(obs)
        obs = observation()
        obs.qpos[0, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            backend.predict(obs)
        obs = observation()
        obs.cameras[backend.camera_fields[0]] = obs.cameras[backend.camera_fields[0]].astype(float)
        with self.assertRaisesRegex(ValueError, "RGB uint8"):
            backend.predict(obs)
        policy.bad_output = True
        with self.assertRaisesRegex(ValueError, "Native decoded"):
            backend.predict(observation())

    def test_checkpoint_mismatch_rejected(self):
        policy = FakePolicy()
        policy.modality_configs["video"].modality_keys.reverse()
        with self.assertRaisesRegex(ValueError, "video order"):
            OpenArmBackend(policy, manifest())
        policy = FakePolicy()
        policy.model.config.use_motion = True
        with self.assertRaisesRegex(ValueError, "addons"):
            OpenArmBackend(policy, manifest())
        policy = FakePolicy()
        policy.processor.state_action_processor.norm_params["general_embodiment"]["action"]["left_arm"]["dim"] = 6
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            OpenArmBackend(policy, manifest())
        wrong = manifest()
        wrong["action_hz"] = 15
        with self.assertRaisesRegex(ValueError, "action_hz"):
            OpenArmBackend(FakePolicy(), wrong)

    def test_rtc_rejected_before_model_load(self):
        for mode in ("trained", "executor"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "supported"):
                    require_supported_rtc(mode)
                with self.assertRaises(SystemExit):
                    parse_args(["--checkpoint-dir", "/nonexistent", "--rtc-mode", mode,
                                "--warmup-steps", "0"])
        parsed = parse_args(["--checkpoint-dir", "/nonexistent", "--rtc-mode", "guided",
                             "--warmup-steps", "0"])
        self.assertEqual(parsed.rtc_mode, "guided")
        parsed = parse_args(["--checkpoint-dir", "/nonexistent", "--rtc-mode", "inpaint",
                             "--rtc-inpaint-ramp-steps", "2", "--warmup-steps", "0"])
        self.assertEqual(parsed.rtc_mode, "inpaint")
        for extra in (["--rtc-mode", "inpaint", "--rtc-guidance-steps", "1"],
                      ["--rtc-mode", "guided", "--rtc-inpaint-ramp-steps", "2"],
                      ["--rtc-mode", "guided", "--rtc-guidance-steps", "5"]):
            with self.assertRaises(SystemExit):
                parse_args(["--checkpoint-dir", "/nonexistent", "--warmup-steps", "0"]+extra)
        with self.assertRaises(SystemExit):
            parse_args(["--checkpoint-dir", "/nonexistent", "--rtc-margin-ms", "nan",
                        "--warmup-steps", "0"])

    def test_guided_dispatch_full_output_skip_and_reset(self):
        from openarm.rtc import RTCRequest

        policy = FakePolicy()
        policy.rtc_inference_mode = "guided"
        class FakeRTC:
            def __init__(self):
                self.mode = "guided"
                self.resets = 0
                self.next = RTCRequest(reset_native=True)
            def prepare(self, obs, started_ns, *, calibration_only=False):
                self.last_started = started_ns
                return self.next
            def reset(self):
                self.resets += 1
            def record_prediction(self, observation, started_ns, elapsed_ns, trace, expected_mode):
                return trace

        rtc = FakeRTC()
        with self.assertRaisesRegex(ValueError, "explicit executor"):
            OpenArmBackend(policy, manifest())
        backend = OpenArmBackend(policy, manifest(), rtc=rtc)
        first = backend.predict(observation())
        self.assertEqual(first.positions.shape, (32, 16))
        self.assertEqual(policy.calls[-1][1], {"session_ids": ["openarm"], "executor_rtc": None})
        self.assertEqual(policy.reset_count, 1)
        target = {"target": np.zeros((1, 32, 64)), "prefix_len": 3,
                  "mask": np.ones((1, 32, 64), dtype=bool)}
        timing = {"based_on_chunk_id": "adopted-1", "action_window_start": 3}
        rtc.next = RTCRequest(options={"executor_rtc": target}, execution=timing,
                              log_data={"mode": "guided", "hard_prefix": False, "used": True})
        guided = backend.predict(observation())
        self.assertEqual(guided.positions.shape, (32, 16))
        self.assertEqual(guided.execution, timing)
        self.assertIs(policy.calls[-1][1]["executor_rtc"], target)
        self.assertFalse(guided.log_data["rtc"]["hard_prefix"])
        count = len(policy.calls)
        rtc.next = RTCRequest(skip=True, log_data={"reason": "delay_exceeds_native_guided_limit"})
        skipped = backend.predict(observation())
        self.assertIsNone(skipped.positions)
        self.assertIsNone(skipped.execution)
        self.assertEqual(len(policy.calls), count)
        backend.reset("trial")
        self.assertEqual(rtc.resets, 1)
        backend.close()
        self.assertEqual(rtc.resets, 2)

    def test_rtc_off_never_sends_executor_target(self):
        obs = observation()
        obs.execution_plan = {"irrelevant": "feedback alone cannot enable guidance"}
        first, second = FakePolicy(), FakePolicy()
        off = OpenArmBackend(first, manifest()).predict(obs)
        ordinary = OpenArmBackend(second, manifest()).predict(observation())
        np.testing.assert_array_equal(off.positions, ordinary.positions)
        self.assertEqual(first.calls[0][1], {"session_ids": ["openarm"]})
        self.assertIsNone(off.execution)

    def test_local_durations_use_monotonic_clock_despite_wall_clock_change(self):
        backend = OpenArmBackend(FakePolicy(), manifest())
        with patch("openarm.backend.time.time_ns", side_effect=[100, 50]), \
             patch("openarm.backend.time.perf_counter_ns", side_effect=[1000, 12000, 20000, 35000]):
            first = backend.predict(observation())
            second = backend.predict(observation())
        self.assertEqual(first.log_data["rtc"]["sampler"]["elapsed_ms"], .011)
        self.assertEqual(second.log_data["rtc"]["sampler"]["elapsed_ms"], .015)

    def test_warmup_reset_and_close(self):
        policy = FakePolicy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warmup.npz"
            obs = observation()
            np.savez(path, qpos=obs.qpos[0], prompt=np.asarray(obs.prompt),
                     **{key: value[0] for key, value in obs.cameras.items()})
            backend = OpenArmBackend(policy, manifest(), warmup_sample=path, warmup_steps=2)
            backend.warmup()
            self.assertEqual(len(policy.calls), 2)
            self.assertEqual(policy.reset_count, 1)
            self.assertFalse(backend.reset("trial"))
            backend.close()
            self.assertEqual(policy.reset_count, 3)
            backend.close()  # Idempotent resource release.
            with self.assertRaisesRegex(RuntimeError, "closed"):
                backend.predict(obs)

    def test_active_warmup_modes_calibration_and_synthetic_state_cleanup(self):
        from openarm.rtc import ExecutionRTC

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warmup.npz"
            obs = observation()
            np.savez(path, qpos=obs.qpos[0], prompt=np.asarray(obs.prompt),
                     **{key: value[0] for key, value in obs.cameras.items()})
            for mode, steps, ramp in (("guided", 1, 0), ("guided", 4, 0), ("inpaint", None, 2)):
                policy = FakePolicy()
                policy.rtc_inference_mode = "guided"
                rtc = ExecutionRTC(policy, mode=mode, guidance_steps=steps, inpaint_ramp_steps=ramp)
                backend = OpenArmBackend(policy, manifest(), rtc=rtc, warmup_sample=path, warmup_steps=2)
                backend.warmup()
                self.assertEqual([row["effective_mode"] for row in backend.warmup_trace], ["none", mode, "none", mode])
                self.assertEqual([row["vjp_calls"] for row in backend.warmup_trace], [0, steps or 0, 0, steps or 0])
                self.assertEqual([row["calibration_only"] for row in backend.warmup_trace],
                                 [False, True, False, True])
                self.assertTrue(all(row["calibration_sample_count"]==1 for row in backend.warmup_trace))
                self.assertTrue(all(row["calibration_scope"]=="latter warm samples" for row in backend.warmup_trace))
                self.assertIsNone(rtc.context)
                self.assertIsNone(rtc.plan)
                self.assertIsNone(rtc.previous_plan)
                self.assertFalse(rtc.pending or rtc.acknowledged)
                calibration = rtc.latency_snapshot()
                self.assertEqual(len(calibration), 2)
                self.assertTrue(all(value > 0 for value in calibration.values()))
                backend.reset("new-trial")
                self.assertEqual(calibration, rtc.latency_snapshot())

    def test_active_warmup_single_sample_is_truthful_and_fallback_rejected(self):
        from openarm.rtc import ExecutionRTC

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warmup.npz"
            obs = observation()
            np.savez(path, qpos=obs.qpos[0], prompt=np.asarray(obs.prompt),
                     **{key: value[0] for key, value in obs.cameras.items()})
            policy = FakePolicy()
            policy.rtc_inference_mode = "guided"
            rtc = ExecutionRTC(policy)
            backend = OpenArmBackend(policy, manifest(), rtc=rtc, warmup_sample=path, warmup_steps=1)
            backend.warmup()
            self.assertTrue(all(row["calibration_scope"]=="single cold sample" for row in backend.warmup_trace))
            original = policy.get_action
            def fallback(native, options):
                actions, info = original(native, options)
                info["rtc"].update(effective_mode="none", vjp_calls=0)
                return actions, info
            policy.get_action = fallback
            with self.assertRaisesRegex(ValueError, "effective path/VJP"):
                backend.warmup()
            self.assertFalse(rtc.pending or rtc.acknowledged)

    def test_real_socket_shm_full_prediction_and_disconnect(self):
        policy = FakePolicy()
        backend = OpenArmBackend(policy, manifest())
        session = ModelSession(backend)
        with tempfile.TemporaryDirectory() as directory:
            obs = observation()
            arrays = {"position": obs.qpos, **obs.cameras}
            payload, fields = bytearray(), {}
            for name, array in arrays.items():
                fields[name] = dict(offset=len(payload), nbytes=array.nbytes,
                                    dtype=array.dtype.str, shape=list(array.shape))
                payload.extend(array.tobytes())
            path = Path(directory) / "input.ring"
            path.write_bytes(payload)
            request = dict(
                transport="shm_ring_v1", chunk_id="cpu-test", record_log=True,
                shm=dict(path=str(path), ring_size=len(payload), slot_size=len(payload),
                         slot_count=1, slot=0, sequence=9, payload_size=len(payload),
                         fields=fields, task_prompt="pillow"),
                metadata=dict(timestamp=123, history_delta_indices="0,-1",
                              action_window_start=3, action_window_size=2),
                reset_reason="trial",
            )
            server, client = socket.socketpair()
            client.settimeout(5)
            errors = []

            def run():
                try:
                    with server, server.makefile("rw", encoding="utf-8") as io:
                        serve_connection(io, session)
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                with client, client.makefile("rw", encoding="utf-8") as io:
                    io.write(json.dumps({"ping": True}) + "\n")
                    io.flush()
                    ready = json.loads(io.readline())
                    self.assertTrue(ready["ready"])
                    self.assertEqual(ready["input_mode"], "copy")
                    self.assertEqual(len(policy.calls), 0)
                    io.write(json.dumps(request) + "\n")
                    io.flush()
                    result = json.loads(io.readline())
                    self.assertNotIn("error", result)
                    self.assertEqual(np.asarray(result["positions"]).shape, (32, 16))
                    self.assertEqual(result["released_input_sequences"], [9])
                    self.assertTrue(result["reset_applied"])
                    self.assertIsNone(result["execution"])
            finally:
                client.close()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(policy.reset_count, 2)  # Trial and disconnect.


if __name__ == "__main__":
    unittest.main()
