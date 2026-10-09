"""CPU numerics with the native codec/sampler and a mock velocity network.

Run only in the isolated model environment: python -m unittest openarm.test_rtc -v
Optional OPENARM_RTC_CHECKPOINT names a saved checkpoint for its real processor
statistics roundtrip. These tests never load model weights or command a robot.
"""

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch
from openarm_policy_runtime import Observation

from openarm.data import modality_config
from openarm.rtc import ExecutionRTC, GROUPS, HORIZON, INTERVAL_NS, PriorCodec, SLICES
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.state_action.state_action_processor import StateActionProcessor
from rldx.model.modules.action_model.rtc import RTCConfig, compute_soft_mask_weights, guidance_scale
from rldx.policy.policy_runtime import PolicyRuntime
from rldx.policy.session_registry import SessionRegistry
from rldx.policy.step_request import StepRequest


def policy_with_processor(sap=None):
    if sap is None:
        def stats(width, horizon=False):
            shape = (HORIZON, width) if horizon else (width,)
            lo = -np.ones(shape)
            hi = np.ones(shape)
            if horizon:
                hi += np.arange(HORIZON)[:, None] / 10
            return dict(min=lo.tolist(), max=hi.tolist(), q01=lo.tolist(), q99=hi.tolist(),
                        mean=np.zeros(shape).tolist(), std=np.ones(shape).tolist())
        statistics = {kind: {key: stats(section.stop-section.start)
                             for key, section in zip(GROUPS, SLICES)} for kind in ("state", "action")}
        statistics["relative_action"] = {key: stats(7, True) for key in ("right_arm", "left_arm")}
        sap = StateActionProcessor({"general_embodiment": modality_config()},
                                   statistics={"general_embodiment": statistics},
                                   use_relative_action=True, clip_outliers=True)
    return NS(embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
              processor=NS(state_action_processor=sap),
              model=NS(action_model=NS(action_dim=64), config=NS(action_horizon=32)),
              rtc_inference_mode="guided", rtc_inference_delay=1, rtc_exec_horizon=16)


def observation(stamp, plan=None, *, prompt="pillow", attempt="attempt-1", trial="trial-1"):
    return Observation(stamp, np.zeros((1, 16), dtype=np.float32), {}, prompt,
                       metadata={"episode_attempt_id": attempt, "inference_trial_id": trial},
                       delta_indices=(0,), execution_plan=plan)


def plan(start, *, chunk="chunk-1", received=None, sample_start=None, count=32):
    return dict(chunk_id=chunk, start_timestamp_ns=start, interval_ns=INTERVAL_NS,
                positions=np.repeat(np.arange(count)[:, None] / 100, 16, axis=1).tolist(),
                sample_chunk_id=chunk, inference_started_timestamp_ns=sample_start or start,
                received_timestamp_ns=received or start + INTERVAL_NS,
                episode_attempt_id="attempt-1")


class PriorCodecTests(unittest.TestCase):
    def check_roundtrip(self, policy):
        original_clipping = policy.processor.state_action_processor.clip_outliers
        codec = PriorCodec(policy)
        current = np.linspace(-10, 10, 16, dtype=np.float32)
        absolute = np.tile(np.linspace(-.3, .4, 16, dtype=np.float32), (32, 1))
        encoded, mask = codec.encode(absolute, current)
        self.assertEqual(encoded.shape, (1, 32, 64))
        self.assertTrue(mask[..., :16].all())
        self.assertFalse(mask[..., 16:].any())
        self.assertGreater(float(np.abs(encoded).max()), 1)
        np.testing.assert_allclose(codec.decode_unclipped(encoded, current), absolute, atol=2e-5)
        changed = current + 3
        encoded_changed, _ = codec.encode(absolute, changed)
        np.testing.assert_allclose(codec.decode_unclipped(encoded_changed, changed), absolute, atol=2e-5)
        np.testing.assert_array_equal(encoded[..., [7, 15]], encoded_changed[..., [7, 15]])
        self.assertFalse(np.array_equal(encoded[..., :7], encoded_changed[..., :7]))
        self.assertEqual(policy.processor.state_action_processor.clip_outliers, original_clipping)
        # Native output decoding clips independently of prior encoder's flag.
        groups = {key: encoded[0, :, section] for key, section in zip(GROUPS, SLICES)}
        state = {key: current[section][None] for key, section in zip(GROUPS, SLICES)}
        clipped = codec.sap.unapply_action(groups, codec.tag, state=state)
        clipped = np.concatenate([clipped[key] for key in GROUPS], axis=-1)
        self.assertGreater(float(np.abs(clipped-absolute).max()), 1)
        print(json.dumps({"rtc_codec": "unclipped_prior_audit", "horizon": 32,
                          "command_dim": 16, "model_dim": codec.dim,
                          "max_roundtrip_error": float(np.abs(codec.decode_unclipped(encoded, current)-absolute).max()),
                          "normalized_abs_max": float(np.abs(encoded).max()),
                          "native_clipped_decoder_max_error": float(np.abs(clipped-absolute).max())}))

    def test_native_horizon_statistics_rebase_and_out_of_range(self):
        self.check_roundtrip(policy_with_processor())

    @unittest.skipUnless(os.environ.get("OPENARM_RTC_CHECKPOINT"), "saved checkpoint not supplied")
    def test_actual_checkpoint_processor_roundtrip(self):
        from rldx.model.core.processing_rldx import RLDXProcessor
        path = Path(os.environ["OPENARM_RTC_CHECKPOINT"])
        processor = RLDXProcessor.from_pretrained(str(path / "processor"),
                                                  model_name=str(path / "backbone"),
                                                  transformers_loading_kwargs={"local_files_only": True})
        self.check_roundtrip(policy_with_processor(processor.state_action_processor))

    @unittest.skipUnless(all(os.environ.get(key) for key in
                            ("OPENARM_RTC_BASE", "OPENARM_RTC_BACKBONE", "OPENARM_RTC_DATASET")),
                         "real training processor assets not supplied")
    def test_actual_training_processor_roundtrip(self):
        from openarm.data import PillowDataset
        from openarm.train import make_processor
        dataset = PillowDataset(Path(os.environ["OPENARM_RTC_DATASET"]))
        processor = make_processor(Path(os.environ["OPENARM_RTC_BASE"]),
                                   Path(os.environ["OPENARM_RTC_BACKBONE"]), dataset)
        self.check_roundtrip(policy_with_processor(processor.state_action_processor))

    def test_degenerate_statistic_cannot_silently_lose_prior(self):
        policy = policy_with_processor()
        params = policy.processor.state_action_processor.norm_params["general_embodiment"]["action"]["right_gripper"]
        params["max"] = params["min"].copy()
        with self.assertRaisesRegex(ValueError, "not invertible"):
            PriorCodec(policy).encode(np.zeros((32, 16)), np.zeros(16))


class TimelineTests(unittest.TestCase):
    origin = 10_000_000_000

    def ready(self):
        rtc = ExecutionRTC(policy_with_processor(), margin_ms=0)
        bootstrap = rtc.prepare(observation(self.origin), self.origin)
        self.assertEqual(bootstrap.options, {"executor_rtc": None})
        self.assertEqual(bootstrap.execution["based_on_chunk_id"], "")
        return rtc

    def test_alignment_short_plan_hold_and_owned_snapshot(self):
        rtc = self.ready()
        source = plan(self.origin, count=2)
        stamp = self.origin + INTERVAL_NS
        result = rtc.prepare(observation(stamp, source), stamp)
        self.assertFalse(result.skip)
        self.assertEqual(result.execution["action_window_start"], 1)
        self.assertEqual(result.execution["takeover_timestamp_ns"], stamp+INTERVAL_NS)
        absolute = rtc.codec.decode_unclipped(result.options["executor_rtc"]["target"], np.zeros(16))
        np.testing.assert_allclose(absolute, .01, atol=1e-7)
        self.assertEqual(result.log_data["held_endpoint_steps"], 31)
        source["positions"][1][0] = 100
        self.assertAlmostEqual(rtc.plan["positions"][1, 0], .01)
        self.assertEqual(rtc.prepare(observation(stamp+1), stamp+1).options, {"executor_rtc": None})

    def test_interpolation_and_previous_plan_pre_handoff(self):
        rtc = self.ready()
        stamp = self.origin+INTERVAL_NS
        rtc.prepare(observation(stamp, plan(self.origin)), stamp)
        second = plan(stamp+2*INTERVAL_NS, chunk="chunk-2", sample_start=stamp,
                      received=stamp+INTERVAL_NS)
        second["positions"] = (np.asarray(second["positions"])+.5).tolist()
        result = rtc.prepare(observation(stamp+INTERVAL_NS, second), stamp+INTERVAL_NS)
        decoded = rtc.codec.decode_unclipped(result.options["executor_rtc"]["target"], np.zeros(16))
        np.testing.assert_allclose(decoded[0], .02, atol=1e-7)
        np.testing.assert_allclose(decoded[1], .5, atol=1e-7)
        half = ExecutionRTC.sample(ExecutionRTC.snapshot(plan(self.origin)),
                                   np.array([self.origin+INTERVAL_NS//2]))
        np.testing.assert_allclose(half, .005, atol=1e-7)

    def test_reset_prompt_trial_attempt_and_stale_feedback(self):
        for key, value in (("prompt", "new"), ("trial", "trial-2"), ("attempt", "attempt-2")):
            rtc = self.ready()
            old = plan(self.origin)
            stamp = self.origin+INTERVAL_NS
            rtc.prepare(observation(stamp, old), stamp)
            changed = observation(stamp+1, old, **{key: value})
            result = rtc.prepare(changed, stamp+1)
            self.assertTrue(result.reset_native)
            self.assertEqual(result.options, {"executor_rtc": None})
            changed.timestamp += 1
            self.assertTrue(rtc.prepare(changed, stamp+2).skip)
        rtc = self.ready()
        rtc.reset()
        result = rtc.prepare(observation(self.origin+1, plan(self.origin)), self.origin+1)
        self.assertEqual(result.options, {"executor_rtc": None})

    def test_invalid_superseded_uncovered_and_expired_inputs(self):
        rtc = self.ready()
        stamp = self.origin+INTERVAL_NS
        latest = plan(self.origin, chunk="latest")
        rtc.prepare(observation(stamp, latest), stamp)
        old = plan(self.origin, chunk="old", received=self.origin+1)
        self.assertEqual(rtc.prepare(observation(stamp, old), stamp).log_data["reason"], "superseded_plan")
        # The inference starts before receipt, while an early result may carry
        # a future takeover. Do not confuse that takeover with inference start.
        misordered = plan(stamp+INTERVAL_NS, received=stamp)
        fresh = self.ready()
        self.assertEqual(fresh.prepare(observation(stamp, misordered), stamp).log_data["reason"],
                         "stale_or_invalid_plan_timing")
        uncovered = dict(misordered, inference_started_timestamp_ns=self.origin)
        self.assertEqual(fresh.prepare(observation(stamp, uncovered), stamp).log_data["reason"],
                         "missing_pre_handoff_coverage")
        expired = rtc.prepare(observation(stamp, latest), stamp+32*INTERVAL_NS)
        self.assertTrue(expired.skip)
        self.assertEqual(expired.log_data["reason"], "delay_exceeds_native_guided_limit")
        latest["positions"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            rtc.prepare(observation(stamp, latest), stamp)

    def test_sample_identity_required_and_delay_skip_logs_measured_latency(self):
        rtc = self.ready()
        stamp = self.origin+INTERVAL_NS
        for sample_id in (None, "", 123):
            source = plan(self.origin)
            source["sample_chunk_id"] = sample_id
            result = rtc.prepare(observation(stamp, source), stamp)
            self.assertTrue(result.skip)
            self.assertEqual(result.log_data["reason"], "missing_sample_identity")
            self.assertEqual(rtc.delay_ns, 0)
        slow_stamp = self.origin+17*INTERVAL_NS
        slow = plan(self.origin, received=slow_stamp)
        result = rtc.prepare(observation(slow_stamp, slow), slow_stamp)
        self.assertTrue(result.skip)
        self.assertEqual(result.log_data["reason"], "delay_exceeds_native_guided_limit")
        self.assertAlmostEqual(result.log_data["delay_ms"], 17*INTERVAL_NS/1e6)

    def test_rejected_bootstrap_feedback_has_no_adopted_prior(self):
        rtc = self.ready()
        stamp = self.origin+INTERVAL_NS
        rtc.prepare(observation(stamp, plan(self.origin)), stamp)
        second = plan(stamp, chunk="second", sample_start=stamp, received=stamp+INTERVAL_NS)
        rtc.prepare(observation(stamp+INTERVAL_NS, second), stamp+INTERVAL_NS)
        self.assertIsNotNone(rtc.plan)
        self.assertIsNotNone(rtc.previous_plan)
        stamp += 2*INTERVAL_NS
        rejected = dict(chunk_id="", positions=[], sample_chunk_id="rejected",
                        inference_started_timestamp_ns=stamp-INTERVAL_NS, received_timestamp_ns=stamp)
        result = rtc.prepare(observation(stamp, rejected), stamp)
        self.assertFalse(result.skip)
        self.assertEqual(result.options, {"executor_rtc": None})
        self.assertEqual(result.log_data["reason"], "no_adopted_plan")
        self.assertEqual(result.execution["based_on_chunk_id"], "")
        self.assertIsNone(rtc.plan)
        self.assertIsNone(rtc.previous_plan)
        future = plan(stamp+INTERVAL_NS, chunk="future", sample_start=stamp-INTERVAL_NS, received=stamp)
        self.assertEqual(rtc.prepare(observation(stamp, future), stamp).log_data["reason"],
                         "missing_pre_handoff_coverage")


class NativeRuntimeTests(unittest.TestCase):
    def runtime(self):
        return PolicyRuntime(model=NS(device="cpu", config=NS(action_horizon=32),
                                      action_model=NS(action_dim=64)), processor=None,
                             modality_configs={}, embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
                             collate_fn=None, language_key="task", registry=SessionRegistry(),
                             infer_lock=None, use_memory=False, rtc_inference_mode="guided",
                             rtc_inference_delay=1, rtc_exec_horizon=16, rtc_enabled=True)

    def test_explicit_targets_and_bootstrap_never_read_cache(self):
        runtime = self.runtime()
        runtime.registry.load_rtc_prefix = lambda *args: self.fail("cached prefix read")
        runtime.registry.load_rtc_postfix_target = lambda *args: self.fail("cached postfix read")
        target, mask = PriorCodec(policy_with_processor()).encode(np.ones((32, 16)), np.zeros(16))
        for value in (None, dict(target=target, mask=mask, prefix_len=3)):
            request = StepRequest({}, 1, sids=["test"], extras={"executor_rtc": value})
            collated = {}
            active = runtime._inject_rtc_prefix(request, collated, 1, None)
            self.assertEqual(active, [])  # step() consequently does not save its prediction.
            if value is None:
                self.assertEqual(collated, {})
            else:
                torch.testing.assert_close(collated["action_prefix"], torch.tensor(target[:, :3]).bfloat16())
                torch.testing.assert_close(collated["action_postfix_target"], torch.tensor(target[:, 3:]).bfloat16())
                self.assertFalse(collated["action_guidance_mask"][..., 16:].any())
        with self.assertRaisesRegex(ValueError, "guided"):
            runtime.rtc_inference_mode = "trained"
            runtime._inject_rtc_prefix(request, {}, 1, None)

    def test_default_disabled_path_and_guided_grad_context(self):
        runtime = self.runtime()
        runtime._rtc_enabled = False
        collated = {"original": 1}
        self.assertEqual(runtime._inject_rtc_prefix(StepRequest({}, 1), collated, 1, None), [])
        self.assertEqual(collated, {"original": 1})
        with runtime._select_inference_context(True, "guided"), torch.enable_grad():
            x = torch.tensor([2.], requires_grad=True)
            self.assertEqual(torch.autograd.grad(x*x, x)[0].item(), 4)

    def test_legacy_cached_guidance_unchanged_without_executor_option(self):
        runtime = self.runtime()
        cached = torch.arange(32*64, dtype=torch.float32).reshape(1, 32, 64)
        runtime.registry.save_rtc_batch(["test"], cached)
        collated = {}
        active = runtime._inject_rtc_prefix(StepRequest({}, 1, sids=["test"]), collated, 1, None)
        self.assertEqual(active, ["test"])
        self.assertEqual(collated["rtc_prefix_len"], 1)
        torch.testing.assert_close(collated["action_prefix"], cached[:, 16:17].bfloat16())
        torch.testing.assert_close(collated["action_postfix_target"], cached[:, 17:].bfloat16())
        self.assertNotIn("action_guidance_mask", collated)


def native_sampler():
    """Compile the actual sampler method without importing the heavyweight VLM.

    Its velocity network below is a differentiable CPU fixture. No copied
    sampler algorithm or checkpoint-level inference claim is involved.
    """
    path = Path(__file__).parents[1] / "rldx/model/core/rldx.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RLDXActionModel")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "get_action_with_features")
    namespace = dict(torch=torch, BatchFeature=lambda data: data, _print=lambda *args: None,
                     compute_soft_mask_weights=compute_soft_mask_weights, guidance_scale=guidance_scale)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


class NativeSamplerTests(unittest.TestCase):
    def run_sampler(self, target, mask="omitted", mode="guided"):
        d, dim = 3, 64
        physics = NS(prepare_inference=lambda *args: NS(attn_mask=None),
                     build_tokens=lambda *args: None, update_state=lambda state, *args: state)
        head = NS(config=NS(action_horizon=32, add_pos_embed=False), action_dim=dim,
                  _rtc=RTCConfig(inference_mode=mode, inference_delay=1,
                                 inference_exec_horizon=16), physics=physics,
                  num_inference_timesteps=3, action_encoder=lambda x, *args: x,
                  action_decoder=lambda x, *args: x,
                  model=lambda hidden_states, **kw: .1*hidden_states+.03*hidden_states.flip(-1))
        inputs = dict(action_prefix=target[:, :d], action_postfix_target=target[:, d:], rtc_prefix_len=d)
        if isinstance(mask, torch.Tensor):
            inputs["action_guidance_mask"] = mask
        torch.manual_seed(7)
        return native_sampler()(head, torch.zeros(1, 1, dim), torch.zeros(1, 1, dim),
                                torch.zeros(1, dtype=torch.long), {}, inputs)["action_pred"]

    def test_omitted_mask_native_path_equal_all_true_and_padding_ignored(self):
        target = torch.ones(1, 32, 64)
        native = self.run_sampler(target)
        torch.testing.assert_close(native, self.run_sampler(target, torch.ones_like(target, dtype=torch.bool)),
                                   rtol=0, atol=0)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[..., :16] = True
        changed = target.clone()
        changed[..., 16:] = 100
        torch.testing.assert_close(self.run_sampler(target, mask), self.run_sampler(changed, mask), rtol=0, atol=0)
        self.assertFalse(torch.equal(native, self.run_sampler(changed)))

    def test_native_soft_prefix_ramp_free_and_not_hard_inpaint(self):
        W = compute_soft_mask_weights(32, 3, 16, device=torch.device("cpu"), dtype=torch.float32)[0]
        torch.testing.assert_close(W[:3], torch.ones(3))
        self.assertTrue(((W[3:16] > 0) & (W[3:16] < 1)).all())
        self.assertTrue((W[3:15] > W[4:16]).all())
        torch.testing.assert_close(W[16:], torch.zeros(16))
        target = torch.ones(1, 32, 64)
        guided, plain = self.run_sampler(target), self.run_sampler(target, mode="none")
        self.assertFalse(torch.equal(guided[:, :3], target[:, :3]))
        self.assertFalse(torch.equal(guided[:, :3], plain[:, :3]))
        self.assertFalse(torch.equal(guided[:, 3:16], plain[:, 3:16]))
        torch.testing.assert_close(guided[:, 16:], plain[:, 16:], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
