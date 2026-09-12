---
license: apache-2.0
language:
- en
- zh
- ru
pipeline_tag: text-generation
tags:
- fquant
- adaptive-quantization
- mixed-precision
- hadamard
- minicpm5
base_model: openbmb/MiniCPM5-2B
---

# MiniCPM5-2B FQuant adaptive research release

This is an experimental calibration-aware release of `openbmb/MiniCPM5-2B`.
It is intended for PyTorch and Transformers evaluation with the custom files in
this directory. The artifact was built with deterministic Rademacher signs,
block Walsh-Hadamard rotations, activation-weighted scale selection, weighted
randomized SVD, robust L0-L3 layer diagnostics, and a mixed-precision policy.

The current quality probe stores 234 high-sensitivity projections in BF16 and
60 remaining MLP gate/up projections in groupwise INT8. The serialized model is
approximately 4.02 GiB versus 4.69 GiB for the BF16 source, a 1.16x size ratio.
This is a quality-oriented intermediate point; it deliberately trades some
compression for lower logit drift. The release report records the exact layer
rank map and calibration moments.

## Update — 2026-09-12

The local release was refreshed after a numerical review of the loader and
KV-BSS path. The update validates GQA head divisibility and attention-mask
shapes, keeps fully masked attention rows finite, initializes scratch models
deterministically, and adds cache-versus-uncached logit parity checks. The
current repository test suite passes **6 tests**. The model artifact itself is
unchanged by these runtime checks; its calibration report remains the source
of truth for the measured quality probe above.

## Load

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./MiniCPM5-2B-Hadamard-GSQ-adaptive",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "./MiniCPM5-2B-Hadamard-GSQ-adaptive",
    trust_remote_code=True,
)
```

The custom loader is required because the weights use FQuant tensor names
(`weight_bf16`, `qweight_int8`, `scales_int8`, `qweight_packed`, `scales`, and
low-rank factors) rather than a standard Transformers linear weight.

## Validation snapshot

On three fixed English, Chinese, and Russian prompts stored in
`adaptive_bifurcation_report.json`, averaged by valid token position and
compared with the raw BF16 model and the previous GSQ release:

| Metric | Previous GSQ | Adaptive hybrid |
| --- | ---: | ---: |
| Relative logit L2 | 0.619 | 0.545 |
| Logit cosine | 0.791 | 0.838 |
| KL from BF16 per valid token | 2.14 | 1.70 |
| Top-1 agreement | 35.3% | 29.5% |

This is a short three-prompt probe, not a benchmark or a claim of general
quality improvement. Run a held-out task evaluation before treating the
artifact as a default release.

## llama.cpp status

Stock llama.cpp cannot load this artifact directly yet. A compatible path will
need a converter and a small backend extension for the paired input rotation,
mixed BF16/INT8/INT4 tensors, and low-rank residual factors. Folding the whole
operation into ordinary GGUF weights would remove the compression advantage,
so the llama.cpp adapter is intentionally deferred until the PyTorch quality
tests are complete.
