import torch

from configuration_minicpm_hadamard import MiniCPMHadamardConfig
from modeling_minicpm_hadamard import (
    DenseBF16Linear,
    Int8Linear,
    MiniCPMHadamardForCausalLM,
    rademacher_signs,
    resolve_rotation_signs,
)


def test_empty_nonpersistent_sign_buffer_is_recovered():
    empty = torch.zeros(64)
    x = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
    recovered = resolve_rotation_signs(empty, 64, 1729, x)
    assert torch.equal(recovered.float(), rademacher_signs(64, 1729))
    assert bool(recovered.any())


def test_mixed_precision_modules_forward_together():
    config = MiniCPMHadamardConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1000,
        rotation_mode="rademacher_hadamard",
        dense_tensor_names=["model.layers.0.self_attn.q_proj"],
        int8_tensor_names=[
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
        ],
        layer_rank_map={"0": 16, "1": 24},
    )
    model = MiniCPMHadamardForCausalLM(config).eval()
    assert isinstance(model.model.layers[0].self_attn.q_proj, DenseBF16Linear)
    assert isinstance(model.model.layers[1].mlp.gate_proj, Int8Linear)
    with torch.no_grad():
        output = model(torch.randint(0, 1000, (1, 3)), use_cache=False)
    assert output.logits.shape == (1, 3, 1000)
    assert bool(torch.isfinite(output.logits).all())
