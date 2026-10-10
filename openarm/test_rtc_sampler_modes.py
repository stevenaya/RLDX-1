"""CPU checks of the actual native sampler with small deterministic velocities.

Run in a registered isolated model job; no checkpoint, optimizer or robot is used.
The N1.7 reference is 6a5a17e68993f80c221f400f7e11690e3400d1c5:
prior injection and rate-zero ramp, adapted to RLDX's unchanged tau 0->1 loop.
"""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from rldx.model.modules.action_model.rtc import RTCConfig, compute_soft_mask_weights, guidance_scale
from rldx.policy.policy_runtime import PolicyRuntime
from rldx.policy.step_request import StepRequest


def native_sampler():
    path = Path(__file__).parents[1] / "rldx/model/core/rldx.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RLDXActionModel")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "get_action_with_features")
    namespace = dict(torch=torch, BatchFeature=lambda data: data, _print=lambda *args: None,
                     compute_soft_mask_weights=compute_soft_mask_weights, guidance_scale=guidance_scale)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


class SamplerModesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def sample(self, mode="inpaint", *, ramp=0, steps=None, mask=None, target=None,
               dtype=torch.float64, nonlinear=False, schedule=None, omit_mode=False):
        self.encodings = []
        target = target if target is not None else torch.linspace(-.7, .7, 32*64).reshape(1, 32, 64)
        if mask is None:
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[..., :16] = True

        def encode(x, t, embodiment):
            self.encodings.append((x.detach().clone(), t.detach().clone()))
            return x

        physics = NS(prepare_inference=lambda *args: NS(attn_mask=None),
                     build_tokens=lambda *args: None, update_state=lambda state, *args: state)
        head = NS(config=NS(action_horizon=32, add_pos_embed=False), action_dim=64,
                  _rtc=RTCConfig(inference_mode="guided", inference_delay=1,
                                 inference_exec_horizon=16, jacobian_steps_only=3),
                  physics=physics, num_inference_timesteps=4, denoising_timesteps=schedule,
                  action_encoder=encode, action_decoder=lambda x, *args: x,
                  model=lambda hidden_states, **kw: (
                      .1*hidden_states+.03*hidden_states.flip(-1) if nonlinear
                      else hidden_states*0+1))
        inputs = {} if mode == "none" else dict(
            action_prefix=target[:, :4], action_postfix_target=target[:, 4:],
            rtc_prefix_len=4, action_guidance_mask=mask,
            rtc_guidance_steps=steps, rtc_ramp_steps=ramp)
        if inputs and not omit_mode:
            inputs["rtc_mode"] = mode
        self.head = head
        torch.manual_seed(20261010)
        return native_sampler()(head, torch.zeros(1, 1, 64, dtype=dtype),
                                torch.zeros(1, 1, 64, dtype=dtype),
                                torch.zeros(1, dtype=torch.long), {}, inputs)

    def test_inpaint_prefix_ramp_free_and_unmodified_token_times(self):
        target = torch.full((1, 32, 64), .125, dtype=torch.float64)
        with patch.object(torch.autograd, "grad", side_effect=AssertionError("inpaint called VJP")):
            result = self.sample(target=target, ramp=12)
        output, trace = result["action_pred"], result["rtc_trace"]
        torch.manual_seed(20261010)
        noise = torch.randn((1, 32, 64), dtype=torch.float64)
        strength = torch.arange(12, dtype=torch.float64)/12
        torch.testing.assert_close(output[:, :4, :16], target[:, :4, :16], rtol=0, atol=0)
        torch.testing.assert_close(output[:, 4:16, :16], target[:, 4:16, :16]+strength[None, :, None])
        torch.testing.assert_close(output[:, 16:, :16], noise[:, 16:, :16]+1)
        torch.testing.assert_close(output[..., 16:], noise[..., 16:]+1)
        for i, (x, token_time) in enumerate(self.encodings):
            torch.testing.assert_close(x[:, :5, :16], target[:, :5, :16], rtol=0, atol=0)
            torch.testing.assert_close(token_time, torch.full_like(token_time, i/4), rtol=0, atol=0)
        self.assertEqual(trace, dict(effective_mode="inpaint", vjp_calls=0, denoising_steps=4,
                                    guidance_steps=0, prefix_len=4, ramp_steps=12,
                                    timesteps=[0., .25, .5, .75, 1.]))

    def test_zero_ramp_and_nonuniform_integration(self):
        target = torch.full((1, 32, 64), .125, dtype=torch.float64)
        result = self.sample(target=target, ramp=0, schedule=[0., .1, .35, .9])
        torch.testing.assert_close(result["action_pred"][:, :4, :16], target[:, :4, :16], rtol=0, atol=0)
        self.assertEqual(result["rtc_trace"]["timesteps"], [0., .1, .35, .9, 1.])
        # The first free point retains initial noise and all Euler increments.
        torch.manual_seed(20261010)
        noise = torch.randn((1, 32, 64), dtype=torch.float64)
        torch.testing.assert_close(result["action_pred"][:, 4:, :16], noise[:, 4:, :16]+1)

    def test_masked_target_padding_independence(self):
        target = torch.zeros(1, 32, 64)
        original = self.sample(target=target, ramp=12, nonlinear=True)["action_pred"]
        target[..., 16:] = 1000
        changed = self.sample(target=target, ramp=12, nonlinear=True)["action_pred"]
        torch.testing.assert_close(original, changed, rtol=0, atol=0)

    def test_zero_mask_preserves_ordinary_output(self):
        mask = torch.zeros(1, 32, 64, dtype=torch.bool)
        ordinary = self.sample("none", nonlinear=True)["action_pred"]
        for mode in ("inpaint", "guided"):
            actual = self.sample(mode, mask=mask, nonlinear=True)["action_pred"]
            torch.testing.assert_close(actual, ordinary, rtol=0, atol=0)

    def test_guidance_step_counts_and_native_default(self):
        default = self.sample("guided", nonlinear=True, omit_mode=True)
        for count in (1, 3, 4):
            with patch.object(torch.autograd, "grad", wraps=torch.autograd.grad) as grad:
                result = self.sample("guided", steps=count, nonlinear=True)
            self.assertEqual(grad.call_count, count)
            self.assertEqual(result["rtc_trace"]["vjp_calls"], count)
            self.assertEqual(result["rtc_trace"]["guidance_steps"], count)
            self.assertEqual(result["rtc_trace"]["denoising_steps"], 4)
            self.assertEqual(self.head._rtc.jacobian_steps_only, 3)
            if count == 3:
                torch.testing.assert_close(result["action_pred"], default["action_pred"], rtol=0, atol=0)

    def test_bfloat16_preserves_quantized_not_original_prior(self):
        target = torch.full((1, 32, 64), .123456, dtype=torch.float32)
        result = self.sample(target=target, dtype=torch.bfloat16)["action_pred"]
        torch.testing.assert_close(result[:, :4, :16], target[:, :4, :16].bfloat16(), rtol=0, atol=0)
        self.assertGreater(float((result[:, :4, :16].float()-target[:, :4, :16]).abs().max()), 0)

    def test_invalid_step_and_ramp_controls_rejected(self):
        with self.assertRaisesRegex(ValueError, "per-request RTC mode"):
            self.sample("trained")  # Never bypass native trained-RTC config validation.
        for steps in (0, 5, -1, 1.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.sample("guided", steps=steps)
        for ramp in (-1, 29, True):
            with self.subTest(ramp=ramp), self.assertRaises(ValueError):
                self.sample(ramp=ramp)


class RuntimeModesTests(unittest.TestCase):
    def runtime(self):
        return PolicyRuntime(model=NS(device="cpu", config=NS(action_horizon=32),
                                      action_model=NS(action_dim=64, num_inference_timesteps=4)),
                             processor=None, modality_configs={}, embodiment_tag=None,
                             collate_fn=None, language_key="task", registry=None,
                             infer_lock=None, use_memory=False, rtc_inference_mode="guided",
                             rtc_inference_delay=1, rtc_exec_horizon=16, rtc_enabled=True)

    def inject(self, **settings):
        prior = dict(target=torch.zeros(1, 32, 64), mask=torch.zeros(1, 32, 64, dtype=torch.bool),
                     prefix_len=4, **settings)
        collated = {}
        self.runtime()._inject_executor_rtc(StepRequest({}, 1, extras={"executor_rtc": prior}), collated, 1)
        return collated

    def test_per_request_dispatch_without_config_mutation(self):
        for mode, steps, ramp in (("guided", 1, 0), ("guided", 4, 0), ("inpaint", None, 12)):
            collated = self.inject(mode=mode, guidance_steps=steps, ramp_steps=ramp)
            self.assertEqual(collated["rtc_mode"], mode)
            self.assertEqual(collated["rtc_guidance_steps"], steps)
            self.assertEqual(collated["rtc_ramp_steps"], ramp)
        self.assertEqual(self.inject()["rtc_mode"], "guided")

    def test_mode_context_and_invalid_options(self):
        with self.runtime()._select_inference_context(True, "inpaint"):
            self.assertTrue(torch.is_inference_mode_enabled())
        with self.runtime()._select_inference_context(True, "guided"):
            self.assertFalse(torch.is_inference_mode_enabled())
        for options in (dict(mode="trained"), dict(mode="inpaint", guidance_steps=1),
                        dict(mode="guided", guidance_steps=0), dict(mode="guided", guidance_steps=5),
                        dict(mode="guided", ramp_steps=2), dict(mode="inpaint", ramp_steps=29)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.inject(**options)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    unittest.main()
