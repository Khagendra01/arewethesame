"""Modal runner for multi-seed replication of the v0.5 mixed adapter.

Trains the mixed adapter across 5 seeds (31, 42, 73, 128, 256) using the
same v0.5 corpus and hyperparameters. Each seed is an independent run.
"""

import subprocess

import modal

REPO_ROOT = "/root/arewethesame"
DATA_ROOT = f"{REPO_ROOT}/experiments/assistant_v05_seed202"
CONDITIONS = ("self", "other", "mixed", "neutral", "spp")
SEEDS = (31, 42, 73, 128, 256)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.51,<5",
        "peft>=0.15,<1",
        "bitsandbytes>=0.45",
        "accelerate>=1.2",
        "hf_transfer>=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("scripts", remote_path=f"{REPO_ROOT}/scripts")
    .add_local_dir(
        "experiments/assistant_v05_seed202/training",
        remote_path=f"{DATA_ROOT}/training",
    )
    .add_local_file(
        "experiments/assistant_v05_seed202/sha256.json",
        remote_path=f"{DATA_ROOT}/sha256.json",
    )
)

app = modal.App("arewethesame-replicate-seeds", image=image)
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
def train_one(condition: str, seed: int):
    cmd = [
        "python",
        "-u",
        f"{REPO_ROOT}/scripts/train_condition_lora.py",
        "--condition",
        condition,
        "--conditions",
        *CONDITIONS,
        "--data-root",
        DATA_ROOT,
        "--output-root",
        f"/outputs/lora_v05_replication/s{seed}",
        "--seed",
        str(seed),
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_vol.commit()
    print(f"===== done {condition} seed={seed} =====", flush=True)
    return f"{condition}_s{seed}"


@app.local_entrypoint()
def main(condition: str = "mixed", seed: int = 0):
    if seed:
        print(train_one.remote(condition, seed), flush=True)
        return
    for s in SEEDS:
        print(f"completed: {train_one.remote(condition, s)}", flush=True)
