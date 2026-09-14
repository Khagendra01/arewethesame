"""Modal GPU runner for the v0.4 history-dependent corpus.

Trains each matched adapter as an independent function invocation and commits
the output volume after every condition, so progress survives client or
container interruptions. `main` maps the conditions in parallel.
"""

import subprocess

import modal

REPO_ROOT = "/root/arewethesame"
DATA_ROOT = f"{REPO_ROOT}/experiments/assistant_v04_seed101"
CONDITIONS = ("neutral", "self", "other", "spp")

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
        "experiments/assistant_v04_seed101/training",
        remote_path=f"{DATA_ROOT}/training",
    )
    .add_local_file(
        "experiments/assistant_v04_seed101/sha256.json",
        remote_path=f"{DATA_ROOT}/sha256.json",
    )
)

app = modal.App("arewethesame-v04-qlora", image=image)
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
def train_one(condition: str):
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
        "/outputs/lora_v04_seed101",
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_vol.commit()
    print(f"===== done {condition} =====", flush=True)
    return condition


@app.local_entrypoint()
def main(condition: str = ""):
    targets = [condition] if condition else list(CONDITIONS)
    for target in targets:
        result = train_one.remote(target)
        print(f"completed: {result}", flush=True)
