"""Run the v0.6 identity-controlled hard evaluation on Modal.

Examples:
    modal run modal_eval_v06.py --arch qwen
    modal run modal_eval_v06.py --arch mistral
    modal run modal_eval_v06.py --arch analyze
    modal run modal_eval_v06.py --arch all

No retraining is performed. Existing mixed adapters in the
`arewethesame-pilot-seed31` volume are reused.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import modal

REPO_ROOT = "/root/arewethesame"
SEEDS = (31, 42, 73, 128, 256)
BENCH_DIR = "/tmp/locked_v06_identity_controlled"
ITEMS = f"{BENCH_DIR}/items.jsonl"
EXPECTED_ITEMS_SHA256 = "153e2af5c2e5578ae54094e8d7d43ca8536b5976a27901a56f086b42ba581aac"
OUT_ROOT = "/outputs/eval_v06_identity_controlled"

ARCH = {
    "qwen": {
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter_template": "/outputs/lora_v05_replication/s{seed}/mixed/best_adapter",
        "batch_size": 12,
    },
    "mistral": {
        "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
        "adapter_template": "/outputs/lora_mistral7b/s{seed}/mixed/best_adapter",
        "batch_size": 8,
    },
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.51,<5",
        "peft>=0.15,<1",
        "bitsandbytes>=0.45",
        "accelerate>=1.2",
        "sentencepiece>=0.2",
        "numpy>=1.26",
        "hf_transfer>=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("scripts", remote_path=f"{REPO_ROOT}/scripts")
    .add_local_dir("eval", remote_path=f"{REPO_ROOT}/eval")
)

app = modal.App("arewethesame-v06-identity-controlled", image=image)
out_vol = modal.Volume.from_name("arewethesame-pilot-seed31", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=21600,
    volumes={
        "/outputs": out_vol,
        "/root/.cache/huggingface": hf_vol,
    },
)
def evaluate_arch(arch: str):
    if arch not in ARCH:
        raise ValueError(f"unknown architecture: {arch}")
    cfg = ARCH[arch]
    arch_out = f"{OUT_ROOT}/{arch}"

    build_cmd = [
        "python", "-u",
        f"{REPO_ROOT}/scripts/build_locked_v06_identity_controlled.py",
        "--out", BENCH_DIR,
        "--per-family", "120",
    ]
    print("+ " + " ".join(build_cmd), flush=True)
    subprocess.run(build_cmd, check=True, cwd=REPO_ROOT)
    digest = hashlib.sha256(Path(ITEMS).read_bytes()).hexdigest()
    if digest != EXPECTED_ITEMS_SHA256:
        raise RuntimeError(f"v0.6 benchmark hash mismatch: {digest} != {EXPECTED_ITEMS_SHA256}")
    print(f"benchmark hash verified: {digest}", flush=True)

    jobs = [("base", None)] + [
        (f"mixed_s{seed}", cfg["adapter_template"].format(seed=seed))
        for seed in SEEDS
    ]
    for label, adapter in jobs:
        cmd = [
            "python", "-u",
            f"{REPO_ROOT}/scripts/eval_v06_identity_controlled.py",
            "--base-model", cfg["base_model"],
            "--label", label,
            "--items", ITEMS,
            "--output", f"{arch_out}/{label}.json",
            "--batch-size", str(cfg["batch_size"]),
            "--bootstrap", "2000",
        ]
        if adapter is not None:
            cmd += ["--adapter", adapter]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        out_vol.commit()
    print(f"{arch} v0.6 evaluation complete", flush=True)


@app.function(
    timeout=1800,
    volumes={"/outputs": out_vol},
)
def analyze_results():
    cmd = [
        "python", "-u",
        f"{REPO_ROOT}/scripts/analyze_v06_multiseed.py",
        "--root", OUT_ROOT,
        "--output", f"{OUT_ROOT}/summary.json",
        "--bootstrap", "5000",
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_vol.commit()
    print(f"summary: {OUT_ROOT}/summary.json and summary.md", flush=True)


@app.local_entrypoint()
def main(arch: str = "all"):
    arch = arch.lower().strip()
    if arch == "qwen":
        evaluate_arch.remote("qwen")
    elif arch == "mistral":
        evaluate_arch.remote("mistral")
    elif arch == "analyze":
        analyze_results.remote()
    elif arch == "all":
        evaluate_arch.remote("qwen")
        evaluate_arch.remote("mistral")
        analyze_results.remote()
    else:
        raise ValueError("--arch must be qwen, mistral, analyze, or all")
