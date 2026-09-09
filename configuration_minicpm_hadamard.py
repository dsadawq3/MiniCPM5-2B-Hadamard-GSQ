"""MiniCPMHadamardConfig — explicit bifurcation_rank added."""
from transformers.configuration_utils import PretrainedConfig


class MiniCPMHadamardConfig(PretrainedConfig):
    model_type = "minicpm_hadamard"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=130560,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=[1, 130073],
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        bits=4,
        group_size=64,
        hadamard_block_size=128,
        residual_rank=16,
        bifurcation_rank=24,
        k_proj_rank=32,
        tau_focus=1.10,
        haze_floor_margin=12.0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.bits = bits
        self.group_size = group_size
        self.hadamard_block_size = hadamard_block_size
        self.residual_rank = residual_rank
        self.bifurcation_rank = bifurcation_rank
        self.k_proj_rank = k_proj_rank
        self.tau_focus = tau_focus
        self.haze_floor_margin = haze_floor_margin
