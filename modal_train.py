"""Modal GPU runner for arewethesame matched QLoRA pilot (seed 31).

Bakes scripts/ + frozen training splits into the image, reuses the
existing hf-hub-cache volume, writes adapters to a dedicated volume.
First target: neutral smoke test on L4.
"""

import subprocess

import modal

REPO_ROOT = "/root/arewethesame"

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
        "experiments/assistant_v03_pilot_seed31/training",
        remote_path=f"{REPO_ROOT}/experiments/assistant_v03_pilot_seed31/training",
    )
    .add_local_file(
        "experiments/assistant_v03_pilot_seed31/sha256.json",
        remote_path=f"{REPO_ROOT}/experiments/assistant_v03_pilot_seed31/sha256.json",
    )
)

app = modal.App("arewethesame-qlora-seed31", image=image)
out_vol = modal.Volume.from_name("arewethesame-pilot-seed31", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=7200,
    volumes={
        "/outputs": out_vol,
        "/root/.cache/huggingface": hf_vol,
    },
)
def train_condition(condition: str):
    cmd = [
        "python",
        f"{REPO_ROOT}/scripts/train_condition_lora.py",
        "--condition",
        condition,
        "--data-root",
        f"{REPO_ROOT}/experiments/assistant_v03_pilot_seed31",
        "--output-root",
        "/outputs/lora_pilot_seed31",
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_vol.commit()
    print(f"done: {condition}", flush=True)


@app.local_entrypoint()
def main(condition: str = "neutral"):
    train_condition.remote(condition)
