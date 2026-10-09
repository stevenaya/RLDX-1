"""CPU numerics for the current-frame SDPA mask fix; no checkpoint/GPU load.

Run in the isolated model job: python -m unittest openarm.test_attention_mask -v
Genuine video/motion pruning with 4D masks is explicitly outside this patch.
"""
import json
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rldx.model.modules.backbone.layer_wrapper import LayerWrapper


def setUpModule():
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(2)


class AttentionLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        self.seen_mask = attention_mask
        batch, length, dimension = hidden_states.shape
        tokens = hidden_states.reshape(batch, length, 2, dimension // 2).transpose(1, 2)
        output = F.scaled_dot_product_attention(tokens, tokens, tokens, attention_mask, dropout_p=0)
        return output.transpose(1, 2).reshape(batch, length, dimension)


class CaptureLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        self.seen = kwargs
        return hidden_states


def identity_inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(71)
    hidden = torch.randn(2, 12, 8, generator=generator, dtype=dtype)
    ids = torch.full((2, 12), 7, dtype=torch.long)
    ids[0, [2, 4, 6, 8]] = ids[1, [3, 5, 7, 9]] = 99
    padding = torch.ones(2, 12, dtype=torch.bool)
    padding[0, 0] = False
    mask = torch.ones(12, 12, dtype=torch.bool).tril()[None, None] & padding[:, None, None, :]
    return hidden, ids, padding, mask


def manual_attention(hidden, mask):
    batch, length, dimension = hidden.shape
    tokens = hidden.reshape(batch, length, 2, dimension // 2).transpose(1, 2)
    work = tokens.float() if hidden.dtype == torch.bfloat16 else tokens
    scores = work @ work.transpose(-2, -1) / math.sqrt(dimension // 2)
    scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.to(work.dtype)
    valid = torch.isfinite(scores).any(-1, keepdim=True)
    weights = torch.softmax(torch.where(valid, scores, torch.zeros_like(scores)), -1) * valid
    return (weights @ work).transpose(1, 2).reshape_as(hidden).to(hidden.dtype)


class AttentionMaskTests(unittest.TestCase):
    def call(self, hidden, ids, mask):
        layer = AttentionLayer()
        wrapper = LayerWrapper(layer, 4, img_pattern=[99], motion_token=1)
        output, kwargs = wrapper(hidden, ids, attention_mask=mask,
            image_wise_encoding=True, num_views=torch.tensor([4, 4]))
        self.assertIs(layer.seen_mask, mask)
        self.assertIs(kwargs["attention_mask"], mask)
        return output

    def test_identity_boolean_and_additive_outputs_and_gradients(self):
        for dtype in (torch.float64, torch.bfloat16):
            for additive in (False, True):
                with self.subTest(dtype=dtype, additive=additive):
                    source, ids, _, allowed = identity_inputs(dtype)
                    mask = torch.zeros_like(allowed, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min) if additive else allowed
                    actual_input = source.clone().requires_grad_()
                    native_input = source.clone().requires_grad_()
                    reference_input = source.clone().requires_grad_()
                    actual = self.call(actual_input, ids, mask)
                    native = AttentionLayer()(native_input, attention_mask=mask)
                    expected = manual_attention(reference_input, mask)
                    weight = torch.linspace(-1, 1, actual.numel(), dtype=dtype).reshape_as(actual)
                    gradient = torch.autograd.grad((actual * weight).sum(), actual_input, retain_graph=True)[0]
                    native_gradient = torch.autograd.grad((native * weight).sum(), native_input)[0]
                    reference_gradient = torch.autograd.grad((expected * weight).sum(), reference_input, retain_graph=True)[0]
                    # Exact native behavior is preserved even for fully blocked
                    # padding queries. Fused SDPA backward for an all-finfo.min
                    # additive row need not equal literal softmax differentiation.
                    torch.testing.assert_close(actual, native, atol=0, rtol=0)
                    torch.testing.assert_close(gradient, native_gradient, atol=0, rtol=0)
                    valid_query = allowed.any(-1).squeeze(1).unsqueeze(-1)
                    valid_weight = weight * valid_query
                    valid_gradient = torch.autograd.grad((actual * valid_weight).sum(), actual_input)[0]
                    reference_valid_gradient = torch.autograd.grad((expected * valid_weight).sum(), reference_input)[0]
                    tolerance = 0.025 if dtype == torch.bfloat16 else 1e-12
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    torch.testing.assert_close(valid_gradient, reference_valid_gradient, atol=tolerance, rtol=tolerance)
                    self.assertTrue(torch.isfinite(actual).all() and torch.isfinite(gradient).all())
                    print(json.dumps({"check": "current_frame_identity_sdpa", "dtype": str(dtype),
                        "mask": "additive" if additive else "boolean", "mask_object_preserved": True,
                        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "direct_native_output_max_absolute_error": float((actual-native).abs().max()),
                        "direct_native_gradient_max_absolute_error": float((gradient-native_gradient).abs().max()),
                        "output_max_absolute_error": float((actual - expected).abs().max()),
                        "valid_query_loss_gradient_max_absolute_error": float((valid_gradient-reference_valid_gradient).abs().max()),
                        "fully_blocked_padding_queries": int((~valid_query).sum()),
                        "native_vs_manual_all_query_gradient_difference": float((gradient-reference_gradient).abs().max())}))

    def test_native_transformers_mask_and_causal_padding_edges(self):
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        hidden, ids, padding, expected_mask = identity_inputs()
        config = Qwen3VLTextConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=2)
        config._attn_implementation = "sdpa"
        mask = create_causal_mask(config, hidden, padding, torch.arange(12), None)
        torch.testing.assert_close(mask, expected_mask, atol=0, rtol=0)
        baseline = self.call(hidden, ids, mask)
        changed_padding = hidden.clone()
        changed_padding[0, 0] += 1000
        torch.testing.assert_close(self.call(changed_padding, ids, mask)[0, 1:], baseline[0, 1:], atol=0, rtol=0)
        changed_future = hidden.clone()
        changed_future[:, -1] -= 1000
        torch.testing.assert_close(self.call(changed_future, ids, mask)[:, :-1], baseline[:, :-1], atol=0, rtol=0)

    def test_genuine_4d_compression_and_cached_masks_are_rejected(self):
        hidden, ids, _, mask = identity_inputs()
        wrapper = LayerWrapper(CaptureLayer(), 4, img_pattern=[99], motion_token=1)
        with self.assertRaisesRegex(ValueError, "identity selection only"):
            wrapper(hidden, ids, attention_mask=mask)  # Default compresses first through last image.
        with self.assertRaisesRegex(ValueError, "identity selection only"):
            wrapper(hidden, ids, attention_mask=mask, image_wise_encoding=True,
                    num_views=torch.tensor([4, 4]), motion_drop_info={"start": 1, "count": 1})
        with self.assertRaisesRegex(ValueError, "square uncached"):
            wrapper(hidden, ids, attention_mask=mask[..., :-1], image_wise_encoding=True,
                    num_views=torch.tensor([4, 4]))

    def test_legacy_2d_compression_and_motion_drop_unchanged(self):
        source = torch.arange(3 * 8 * 4, dtype=torch.float64).reshape(3, 8, 4)
        ids = torch.full((3, 8), 7, dtype=torch.long)
        ids[0, [2, 5]], ids[1, [4, 6]], ids[2, [4]] = 99, 99, 99
        padding = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1], [1] * 8])
        for motion in (False, True):
            expected_indices = [[0, 1, 2, 5, 6, 7], [0, 1, 2, 3, 4, 6, 7], list(range(8))]
            if motion:
                expected_indices = [[index for index in row if index != 1] for row in expected_indices]
            expected_states, expected_masks = [], []
            for batch, indices in enumerate(expected_indices):
                values, valid = source[batch, indices].clone(), padding[batch, indices].clone()
                if batch < 2:
                    begin, end = ((2, 5), (4, 6))[batch]
                    position = indices.index(begin)
                    values[position], valid[position] = source[batch, begin:end].mean(0), 1
                expected_states.append(values)
                expected_masks.append(valid)
            length = max(len(row) for row in expected_indices)
            expected_states = torch.stack([F.pad(row, (0, 0, length-len(row), 0)) for row in expected_states])
            expected_masks = torch.stack([F.pad(row, (length-len(row), 0)) for row in expected_masks])
            layer = CaptureLayer()
            wrapper = LayerWrapper(layer, 4, img_pattern=[99], motion_token=1)
            output, kwargs = wrapper(source, ids, attention_mask=padding, attention_mask_2=padding,
                motion_drop_info={"start": 1, "count": 1} if motion else None)
            torch.testing.assert_close(output, expected_states, atol=0, rtol=0)
            torch.testing.assert_close(kwargs["attention_mask"], expected_masks, atol=0, rtol=0)
            torch.testing.assert_close(kwargs["attention_mask_2"], expected_masks, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
