"""Modal GPU runner for the locked_v2_contextual factorial evaluation.

Runs scripts/eval_factorial_locked_v2.py on an L4 over the frozen items file and
writes the full model x context results back to the output volume.
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
        "numpy>=1.26",
        "hf_transfer>=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("scripts", remote_path=f"{REPO_ROOT}/scripts")
    .add_local_dir("src", remote_path=f"{REPO_ROOT}/src")
    .add_local_dir("eval", remote_path=f"{REPO_ROOT}/eval")
)

app = modal.App("arewethesame-eval-locked-v2", image=image)
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
def evaluate(
    generation_subset: int = 30,
    limit_items: int = 0,
    adapters_root: str = "/outputs/lora_pilot_seed31",
    output: str = "/outputs/eval_locked_v2/factorial.json",
    models: str = "",
    items: str = f"{REPO_ROOT}/eval/locked_v2_contextual/items.jsonl",
):
    cmd = [
        "python",
        "-u",
        f"{REPO_ROOT}/scripts/eval_factorial_locked_v2.py",
        "--adapters-root",
        adapters_root,
        "--items",
        items,
        "--output",
        output,
        "--bootstrap",
        "2000",
        "--generation-subset",
        str(generation_subset),
    ]
    if models:
        cmd += ["--models", *models.split()]
    if limit_items:
        cmd += ["--limit-items", str(limit_items)]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_vol.commit()
    print("eval v2 done", flush=True)


@app.local_entrypoint()
def main(
    generation_subset: int = 30,
    limit_items: int = 0,
    adapters_root: str = "/outputs/lora_pilot_seed31",
    output: str = "/outputs/eval_locked_v2/factorial.json",
    models: str = "",
    items: str = f"{REPO_ROOT}/eval/locked_v2_contextual/items.jsonl",
):
    evaluate.remote(
        generation_subset=generation_subset,
        limit_items=limit_items,
        adapters_root=adapters_root,
        output=output,
        models=models,
        items=items,
    )
