"""Evaluate the 5 replicated mixed adapters on the OOD held-out templates set."""

import subprocess

import modal

REPO_ROOT = "/root/arewethesame"
SEEDS = (31, 42, 73, 128, 256)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.51,<5",
        "peft>=0.15,<1",
        "bitsandbytes>=0.45",
        "accelerate>=1.2",
        "numpy>=1.26",
        "hf_transfer>=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("scripts", remote_path=f"{REPO_ROOT}/scripts")
    .add_local_dir("src", remote_path=f"{REPO_ROOT}/src")
    .add_local_dir("eval", remote_path=f"{REPO_ROOT}/eval")
)

app = modal.App("arewethesame-eval-ood", image=image)
out_vol = modal.Volume.from_name("arewethesame-pilot-seed31", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=10800,
    volumes={
        "/outputs": out_vol,
        "/root/.cache/huggingface": hf_vol,
    },
)
def evaluate_all_seeds():
    for seed in SEEDS:
        cmd = [
            "python", "-u",
            f"{REPO_ROOT}/scripts/eval_factorial_locked_v2.py",
            "--adapters-root", f"/outputs/lora_v05_replication/s{seed}",
            "--items", f"{REPO_ROOT}/eval/ood_templates_v1/items.jsonl",
            "--output", f"/outputs/eval_ood_v1/s{seed}/factorial.json",
            "--models", "base", "mixed",
            "--contexts", "self", "other",
            "--bootstrap", "2000",
        ]
        print(f"===== seed {seed} =====", flush=True)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_vol.commit()
    print("all seeds evaluated on OOD", flush=True)


@app.local_entrypoint()
def main():
    evaluate_all_seeds.remote()
