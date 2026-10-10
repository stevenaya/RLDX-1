"""CPU SHM/caller checks for the checkpoint harness; no model weights are loaded.

The isolated job must include the pinned caller and runtime src paths on PYTHONPATH.
"""
import json
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from dora_openarm_local_policy_server.session import Session
from dora_openarm_local_policy_server.shm_ring import SharedMemoryRingWriter
from openarm_policy_runtime import ModelSession

from openarm.check_inference import (
    acknowledge_input, arrow_input, completed_formal_checkpoint, parse_args,
    receipt_summary, require_bootstrap, server_command, summary,
)
from openarm.test_backend import CAMERAS, FakePolicy, OpenArmBackend, manifest


def setUpModule():
    import cv2
    import pyarrow as pa
    cv2.setNumThreads(2)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)


class CheckInferenceTests(unittest.TestCase):
    def test_margin_cli_forwarding_preserves_lateness_and_serving_defaults(self):
        from openarm.serve import parse_args as parse_server

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "save-completion.json").write_text(json.dumps(
                {"complete": True, "formal": True, "optimizer_updates": 500}))
            warmup = root / "synthetic.npz"
            warmup.touch()  # CLI validation only; no sample/model is loaded.
            argv = ["--checkpoint-dir", str(checkpoint), "--warmup-sample", str(warmup),
                    "--eval-workspace", temporary, "--output", str(root / "fresh.json"),
                    "--rtc-mode", "guided"]
            with patch.dict(os.environ, RLDX_OUTPUT=temporary, MODEL_LAB_STAGE_KIND="none",
                            MODEL_LAB_MAX_STEPS="0"), patch("openarm.check_inference.job_guard"):
                for options, margin in (([], 10), (["--rtc-margin-ms", "150"], 150)):
                    with self.subTest(margin=margin):
                        args = parse_args(argv + options)
                        command = server_command(args, root / "policy.socket")
                        served = parse_server(command[3:])
                        self.assertEqual(served.rtc_mode, "guided")
                        self.assertEqual(served.rtc_margin_ms, margin)
                        self.assertEqual(served.rtc_max_lateness_ms, 50)
                        self.assertEqual(served.warmup_steps, 1)
            default = parse_server(["--checkpoint-dir", str(checkpoint), "--warmup-steps", "0"])
            self.assertEqual((default.rtc_mode, default.rtc_margin_ms, default.rtc_max_lateness_ms),
                             ("none", 10, 50))

    def test_active_mode_controls_forward_to_server_without_changing_defaults(self):
        from openarm.serve import parse_args as parse_server

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "save-completion.json").write_text(json.dumps(
                {"complete": True, "formal": True, "optimizer_updates": 500}))
            sample = root / "sample.npz"
            sample.touch()
            argv = ["--checkpoint-dir", str(checkpoint), "--warmup-sample", str(sample),
                    "--eval-workspace", temporary, "--output", str(root / "fresh.json"),
                    "--warmup-steps", "2"]
            with patch.dict(os.environ, RLDX_OUTPUT=temporary, MODEL_LAB_STAGE_KIND="none",
                            MODEL_LAB_MAX_STEPS="0"), patch("openarm.check_inference.job_guard"):
                for mode, control, value in (("guided", "rtc-guidance-steps", 1),
                                             ("guided", "rtc-guidance-steps", 4),
                                             ("inpaint", "rtc-inpaint-ramp-steps", 12)):
                    with self.subTest(mode=mode, value=value):
                        args = parse_args(argv + ["--rtc-mode", mode, "--" + control, str(value)])
                        served = parse_server(server_command(args, root / "socket")[3:])
                        self.assertEqual(served.rtc_mode, mode)
                        self.assertEqual(getattr(served, control.replace("-", "_")), value)
                        self.assertEqual(served.warmup_steps, 2)
                        self.assertEqual((served.rtc_margin_ms, served.rtc_max_lateness_ms), (10, 50))

    def test_cross_mode_controls_are_rejected_before_execution(self):
        argv = ["--checkpoint-dir", "/checkpoint", "--warmup-sample", "/sample",
                "--eval-workspace", "/eval", "--output", "/output"]
        for options in (["--rtc-mode", "inpaint", "--rtc-guidance-steps", "1"],
                        ["--rtc-mode", "guided", "--rtc-inpaint-ramp-steps", "12"],
                        ["--rtc-mode", "none", "--rtc-guidance-steps", "3"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv + options)

    def test_invalid_margin_is_rejected_before_execution(self):
        argv = ["--checkpoint-dir", "/checkpoint", "--warmup-sample", "/sample",
                "--eval-workspace", "/eval", "--output", "/output"]
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit):
                    parse_args(argv + ["--rtc-margin-ms", value])
                self.assertIn("finite nonnegative milliseconds", errors.getvalue())

    def test_guided_receipt_deadline_measurements(self):
        response = {"positions": np.zeros((32, 16)), "interval": 33_333_333,
                    "execution": {"action_origin_timestamp_ns": 1_000_000_000,
                                  "inference_started_timestamp_ns": 1_010_000_000,
                                  "takeover_timestamp_ns": 1_200_000_000,
                                  "max_lateness_ns": 50_000_000},
                    "log_data": {"rtc": {"used": True}}}
        timing = receipt_summary(response, 1_250_000_000)["guided_receipt_timing"]
        self.assertEqual(timing["observation_age_ms"], 10)
        self.assertEqual(timing["inference_start_to_receipt_ms"], 240)
        self.assertEqual(timing["deadline_slack_ms"], 0)
        self.assertTrue(timing["received_by_deadline"])
        self.assertFalse(receipt_summary(response, 1_250_000_001)
                         ["guided_receipt_timing"]["received_by_deadline"])
        response["log_data"]["rtc"]["mode"] = "inpaint"
        inpaint = receipt_summary(response, 1_250_000_000)
        self.assertEqual(inpaint["rtc_receipt_timing"], timing)
        self.assertNotIn("guided_receipt_timing", inpaint)

    def test_only_completed_formal_500_marker_is_accepted(self):
        complete = {"complete": True, "formal": True, "optimizer_updates": 500}
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            marker = checkpoint / "save-completion.json"
            marker.write_text(json.dumps(complete))
            self.assertEqual(completed_formal_checkpoint(checkpoint), complete)
            for change in ({"complete": False}, {"formal": False}, {"optimizer_updates": 12},
                           {"optimizer_updates": 499}, {"complete": "true"}):
                with self.subTest(change=change):
                    marker.write_text(json.dumps(dict(complete, **change)))
                    with self.assertRaisesRegex(ValueError, "complete formal 500"):
                        completed_formal_checkpoint(checkpoint)

    def test_bootstrap_requires_full_output_and_no_prior(self):
        clean = {"prefill": False, "execution": {"based_on_chunk_id": ""},
                 "log_data": {"rtc": {"used": False}}}
        require_bootstrap(clean, "reset", guided=True)
        require_bootstrap({"prefill": False}, "ordinary reset", guided=False)
        for change in ({"prefill": True}, {"execution": None},
                       {"execution": {"based_on_chunk_id": "old"}},
                       {"log_data": {"rtc": {"used": True}}}):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                require_bootstrap(dict(clean, **change), "reset", guided=True)

    def test_native_shm_writer_caller_consumption_and_full_outputs(self):
        sample = {"qpos": np.arange(16, dtype=np.float32), "prompt": np.asarray("synthetic")}
        sample.update({"camera_" + key: np.full((3, 4, 3), index, dtype=np.uint8)
                       for index, key in enumerate(CAMERAS)})
        observation, metadata = arrow_input(sample)
        metadata["timestamp"] = time.time_ns()
        backend = OpenArmBackend(FakePolicy(), manifest())
        server, caller = ModelSession(backend), Session(reset_gap=None, action_window_size=2)
        with tempfile.TemporaryDirectory() as temporary:
            writer = SharedMemoryRingWriter(temporary)
            try:
                written = writer.write(observation, metadata)
                first = caller.prepare(1, metadata, "synthetic")
                first.update(transport="shm_ring_v1", shm=written.descriptor)
                response = server.handle(first)
                acknowledge_input(response, written)
                self.assertEqual(summary(response)["positions_shape"], [32, 16])
                selected = caller.complete(first, response)
                self.assertEqual(len(selected["positions"]), 2)
                caller.sent(selected)
                plan = {"chunk_id": "offline-adopted", "positions": response["positions"]}
                second = caller.prepare(1, dict(metadata, timestamp=time.time_ns()), "synthetic", plan)
                self.assertFalse(second["restart_execution"])
                self.assertIs(second["execution_plan"], plan)
                self.assertIsNone(second["reset_reason"])
            finally:
                server.close()
                caller.close()
                writer.close()
                backend.close()

    def test_missing_release_and_invalid_output_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "acknowledged and released"):
            acknowledge_input({"input_sequence": 2, "released_input_sequences": []},
                              SimpleNamespace(descriptor={"sequence": 2}))
        with self.assertRaisesRegex(AssertionError, "full finite"):
            summary({"positions": np.zeros((2, 16)), "interval": 33_333_333})


if __name__ == "__main__":
    unittest.main()
