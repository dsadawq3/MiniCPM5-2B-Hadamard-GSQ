"""Self-contained numerical checks for the MiniCPM5-2B-Hadamard-GSQ code."""

import unittest
import torch
from configuration_minicpm_hadamard import MiniCPMHadamardConfig
from modeling_minicpm_hadamard import MiniCPMHadamardForCausalLM
from kv_bss import KVBSSAttentionHook


def tiny_config(**overrides):
    values = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=97,
        group_size=8,
        hadamard_block_size=8,
        max_position_embeddings=64,
        use_cache=True,
    )
    values.update(overrides)
    return MiniCPMHadamardConfig(**values)


class MiniCPMNumericalTests(unittest.TestCase):
    def test_architecture_forward_is_finite(self):
        torch.manual_seed(0)
        model = MiniCPMHadamardForCausalLM(tiny_config()).eval()
        input_ids = torch.randint(0, 97, (1, 8))
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
        self.assertEqual(tuple(out.logits.shape), (1, 8, 97))
        self.assertTrue(torch.isfinite(out.logits).all())

    def test_kv_bss_mask_and_gqa_validation(self):
        hook = KVBSSAttentionHook(tau_focus=1.0, haze_floor_margin=100.0)
        query = torch.zeros(1, 2, 2, 4)
        key = torch.zeros(1, 1, 3, 4)
        value = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
        # A 2D keep/pad mask must never route the padded value to the output.
        out = hook(query, key, value, attention_mask=torch.tensor([[1, 1, 0]]))
        expected = value[:, :, :2].mean(dim=2).unsqueeze(2).expand_as(out)
        self.assertTrue(torch.allclose(out, expected))
        # Fully masked rows remain finite and resolve to a zero vector.
        blocked = torch.zeros(1, 3)
        out_blocked = hook(query, key, value, attention_mask=blocked)
        self.assertTrue(torch.isfinite(out_blocked).all())
        self.assertEqual(float(out_blocked.abs().sum()), 0.0)
        with self.assertRaises(ValueError):
            bad_key = torch.zeros(1, 3, 3, 4)
            bad_value = torch.zeros(1, 3, 3, 4)
            hook(query, bad_key, bad_value)

    def test_cached_and_uncached_logits_match(self):
        torch.manual_seed(1)
        model = MiniCPMHadamardForCausalLM(tiny_config()).eval()
        input_ids = torch.tensor([[3, 7, 11, 19, 23]])
        with torch.no_grad():
            full = model(input_ids, use_cache=False).logits
            prefix = model(input_ids[:, :3], use_cache=True)
            suffix = model(
                input_ids[:, 3:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
        self.assertTrue(torch.isfinite(suffix.logits).all())
        torch.testing.assert_close(
            suffix.logits, full[:, 3:], rtol=2e-4, atol=2e-5
        )

    def test_invalid_mask_is_rejected(self):
        hook = KVBSSAttentionHook()
        q = torch.zeros(1, 1, 1, 4)
        k = torch.zeros(1, 1, 2, 4)
        v = torch.zeros(1, 1, 2, 4)
        with self.assertRaises(ValueError):
            hook(q, k, v, attention_mask=torch.ones(1, 3))

    def test_nonfinite_scores_are_contained(self):
        hook = KVBSSAttentionHook(tau_focus=1.0, haze_floor_margin=12.0)
        q = torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16)
        k = torch.zeros(1, 1, 2, 4, dtype=torch.bfloat16)
        v = torch.ones(1, 1, 2, 4, dtype=torch.bfloat16)
        q[0, 0, 0, 0] = float("nan")
        k[0, 0, 1, 0] = float("inf")
        out = hook(q, k, v)
        self.assertTrue(torch.isfinite(out).all())

if __name__ == "__main__":
    unittest.main(verbosity=2)
