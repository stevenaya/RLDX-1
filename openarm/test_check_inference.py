"""CPU SHM/caller checks for the checkpoint harness; no model weights are loaded.

The isolated job must include the pinned caller and runtime src paths on PYTHONPATH.
"""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from dora_openarm_local_policy_server.session import Session
from dora_openarm_local_policy_server.shm_ring import SharedMemoryRingWriter
from openarm_policy_runtime import ModelSession

from openarm.check_inference import (
    acknowledge_input, arrow_input, completed_formal_checkpoint, require_bootstrap, summary,
)
from openarm.test_backend import CAMERAS, FakePolicy, OpenArmBackend, manifest


class CheckInferenceTests(unittest.TestCase):
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
