"""
Self-contained verification test for MiniCPM5-2B-Hadamard-GSQ architecture.
"""

import torch
from configuration_minicpm_hadamard import MiniCPMHadamardConfig
from modeling_minicpm_hadamard import MiniCPMHadamardForCausalLM

def test_architecture():
    print("Testing MiniCPM Hadamard architecture initialization...")
    cfg = MiniCPMHadamardConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1000
    )
    model = MiniCPMHadamardForCausalLM(cfg)
    input_ids = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        out = model(input_ids)
    print(f"Forward pass successful! Logits shape: {out.logits.shape}")
    assert out.logits.shape == (1, 8, 1000)
    print("ALL ARCHITECTURAL SANITY CHECKS PASSED WITH ZERO ERRORS!")

if __name__ == "__main__":
    test_architecture()
