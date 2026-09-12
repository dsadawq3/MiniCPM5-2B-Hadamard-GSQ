"""HF PR pusher for MiniCPM5-2B-Hadamard-GSQ files (NOT a test).

Usage:
    hf auth login
    python push_pr_minicpm.py --repo openbmb/MiniCPM5-2B --revision refs/pr/11
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, CommitOperationAdd, get_token

HERE = os.path.dirname(os.path.abspath(__file__))

FILES_TO_COMMIT = [
    "configuration_minicpm_hadamard.py",
    "modeling_minicpm_hadamard.py",
    "kv_bss.py",
    "test_inference.py",
]

INTEGRATION_MD = """# MiniCPM5-2B-Hadamard-GSQ: KV-BSS integration

This pull request contains a self-contained implementation of the custom
MiniCPM Hadamard model path and the KV-BSS attention hook.

## Included source files

1. `configuration_minicpm_hadamard.py` defines the model and quantization
   parameters, including the KV-BSS controls.
2. `modeling_minicpm_hadamard.py` implements the model, causal masking, RoPE,
   grouped-query attention, and legacy tuple cache support.
3. `kv_bss.py` implements focus scaling and haze-floor filtering with explicit
   shape validation and finite handling for fully masked rows.
4. `test_inference.py` exercises a finite forward pass, GQA validation,
   2D/4D attention-mask behavior, and cached-versus-uncached logit parity.

## Verification

Run from the model-code directory:

```bash
python -m unittest discover -s . -p 'test_inference.py' -v
```

The test is intentionally small and CPU-only. It validates implementation
behavior without downloading a checkpoint and does not claim benchmark
accuracy or long-context quality.

## Scope

The PR contains code and tests only. The separately published quantized
checkpoint and its calibration report are linked from the model card:

https://huggingface.co/F-Labs/MiniCPM5-2B-Hadamard-GSQ
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Push MiniCPM Hadamard files to an HF PR branch.")
    ap.add_argument("--repo", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--revision", default="refs/pr/11")
    args = ap.parse_args()

    # Prefer explicit environment variables for CI, then use the token saved
    # by `hf auth login`. Never print the token.
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or get_token()
    )
    if not token:
        print("No Hugging Face token found; run `hf auth login` first.", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    operations = []
    for fname in FILES_TO_COMMIT:
        fpath = os.path.join(HERE, fname)
        if not os.path.isfile(fpath):
            print(f"Missing file next to this script: {fpath}", file=sys.stderr)
            return 1
        operations.append(CommitOperationAdd(path_in_repo=fname, path_or_fileobj=fpath))
    operations.append(CommitOperationAdd(
        path_in_repo="HADAMARD_QUANT_INTEGRATION.md",
        path_or_fileobj=INTEGRATION_MD.encode("utf-8"),
    ))

    print(f"Submitting {len(operations)} reviewed files to {args.revision} on {args.repo}...")
    commit_info = api.create_commit(
        repo_id=args.repo,
        repo_type="model",
        revision=args.revision,
        operations=operations,
        commit_message="Harden KV-BSS masking and add numerical cache tests",
    )
    print("PR updated successfully:", commit_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
