"""HF PR pusher for MiniCPM5-2B-Hadamard-GSQ files (NOT a test).

Usage:
    $env:HF_TOKEN = "<token>"
    python push_pr_minicpm.py --repo openbmb/MiniCPM5-2B --revision refs/pr/11
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, CommitOperationAdd

HERE = os.path.dirname(os.path.abspath(__file__))

FILES_TO_COMMIT = [
    "configuration_minicpm_hadamard.py",
    "modeling_minicpm_hadamard.py",
    "kv_bss.py",
    "test_inference.py",
]

INTEGRATION_MD = """# MiniCPM5-2B-Hadamard-GSQ: Complete Edge Quantization & KV-BSS Architecture
...
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Push MiniCPM Hadamard files to an HF PR branch.")
    ap.add_argument("--repo", default="openbmb/MiniCPM5-2B")
    ap.add_argument("--revision", default="refs/pr/11")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN env var is not set; refusing to push.", file=sys.stderr)
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

    print(f"Submitting {len(operations)} production files to {args.revision} on {args.repo}...")
    commit_info = api.create_commit(
        repo_id=args.repo,
        repo_type="model",
        revision=args.revision,
        operations=operations,
        commit_message="Add complete production modeling, configuration, test suite, and integration documentation",
    )
    print("PR updated successfully:", commit_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
