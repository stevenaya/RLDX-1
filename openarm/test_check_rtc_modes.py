"""CPU-only evidence-harness checks; execute only in a registered model job."""
import json
import os
from pathlib import Path
import unittest

import numpy as np

from openarm.check_rtc_modes import bounded_short_prior, check_native_trace, regions


class OffsetCodec:
    dim = 64

    def decode_unclipped(self, value, current):
        # The test intentionally has per-offset ranges and state-relative arms.
        widths = np.linspace(1, 2, 32, dtype=np.float32)[:, None]
        result = value[0, :, :16] * widths
        result[:, :7] += current[:7]
        result[:, 8:15] += current[8:15]
        return result


class RTCModeHarnessTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("OPENARM_RTC_CHECKPOINT") and os.environ.get("OPENARM_RTC_WARMUP"),
                         "saved checkpoint metadata and recorded warmup path not supplied")
    def test_actual_checkpoint_short_prior_preflight_without_weights(self):
        from openarm.rtc import PriorCodec
        from openarm.test_rtc import policy_with_processor
        from rldx.model.core.processing_rldx import RLDXProcessor

        checkpoint = Path(os.environ["OPENARM_RTC_CHECKPOINT"])
        processor = RLDXProcessor.from_pretrained(str(checkpoint / "processor"),
            model_name=str(checkpoint / "backbone"),
            transformers_loading_kwargs={"local_files_only": True})
        codec = PriorCodec(policy_with_processor(processor.state_action_processor))
        with np.load(os.environ["OPENARM_RTC_WARMUP"], allow_pickle=False) as sample:
            raw = sample["qpos"].copy()
        current = raw.copy()
        current[:7] += .001
        current[8:15] -= .001
        absolute = bounded_short_prior(codec, raw, current)
        target, mask = codec.encode(absolute, current)
        codec.validate_inpaint(target, current, 4, 12)
        changed = current.copy()
        changed[0] += 10
        bad, _ = codec.encode(absolute, changed)
        with self.assertRaises(ValueError):
            codec.validate_inpaint(bad, changed, 4, 12)
        self.assertTrue(mask[..., :16].all())
        self.assertFalse(mask[..., 16:].any())
        print(json.dumps({"rtc_modes_prior_preflight": "passed", "weights_loaded": False,
            "normalized_abs_max": float(np.abs(target[..., :16]).max()),
            "out_of_range_abs_max": float(np.abs(bad[..., :16]).max())}))

    def test_effective_branch_and_vjp_count_are_required(self):
        trace = {"effective_mode": "guided", "vjp_calls": 3, "denoising_steps": 4}
        check_native_trace(trace, "guided", 3)
        for value in (None, dict(trace, effective_mode="none"), dict(trace, vjp_calls=0),
                      dict(trace, denoising_steps=3)):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                check_native_trace(value, "guided", 3)

    def test_short_plan_holds_endpoint_and_respects_all_offset_ranges(self):
        raw = np.full(16, 50, np.float32)  # Deliberately outside the codec ranges.
        current = np.linspace(-.1, .1, 16, dtype=np.float32)
        codec = OffsetCodec()
        target = bounded_short_prior(codec, raw, current)
        low = codec.decode_unclipped(np.full((1, 32, 64), -.75, np.float32), current)
        high = codec.decode_unclipped(np.full((1, 32, 64), .75, np.float32), current)
        self.assertEqual(target.shape, (32, 16))
        self.assertTrue(np.isfinite(target).all())
        self.assertTrue((target >= low).all())
        self.assertTrue((target <= high).all())
        np.testing.assert_array_equal(target[2:], np.repeat(target[1:2], 30, axis=0))

    def test_region_metrics_separate_frozen_transition_and_free(self):
        left, right = np.zeros((32, 16)), np.zeros((32, 16))
        left[:4] = 1
        left[4:16] = 2
        left[16:] = 3
        result = regions(left, right)
        self.assertEqual([result[key]["max_absolute"] for key in ("frozen", "transition", "free")],
                         [1, 2, 3])
        self.assertNotIn("transition", regions(left, right, prefix=4, end=4))


if __name__ == "__main__":
    unittest.main()
