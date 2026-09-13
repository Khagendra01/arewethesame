"""Modal GPU runner for the locked behavioral evaluation (base + 5 adapters).

Reuses the hf-hub-cache and adapters produced by modal_train.py. Runs the
unmodified scripts/eval_behavioral_locked.py with greedy decoding over the
frozen eval/locked_v1 scenarios and writes results back to the output volume.
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
    .add_local_dir("src", remote_path=f"{REPO_ROOT}/src")
    .add_local_dir("eval", remote_path=f"{REPO_ROOT}/eval")
)

app = modal.App("arewethesame-eval-locked-v1", image=image)
out_vol = modal.Volume.from_name("arewethesame-pilot-seed31", create_if_missing=True)
hf_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=3600,
    volumes={
        "/outputs": out_vol,
        "/root/.cache/huggingface": hf_vol,
    },
)
def evaluate():
    for scoring in ("generate", "logprob"):
        cmd = [
            "python",
            f"{REPO_ROOT}/scripts/eval_behavioral_locked.py",
            "--adapters-root",
            "/outputs/lora_pilot_seed31",
            "--scoring",
            scoring,
            "--output",
            f"/outputs/eval_locked_v1/results_{scoring}.json",
        ]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        out_vol.commit()
    print("eval done", flush=True)


@app.local_entrypoint()
def main():
    evaluate.remote()
