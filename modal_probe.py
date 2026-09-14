"""Modal GPU runner for the seed 31 activation probe.

Runs the probe in-process so exceptions propagate with a real traceback, and
commits the report (or a traceback file) through the mounted output volume.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

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
        "scikit-learn>=1.4",
        "hf_transfer>=0.1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("scripts", remote_path=f"{REPO_ROOT}/scripts")
    .add_local_dir("src", remote_path=f"{REPO_ROOT}/src")
    .add_local_dir("eval", remote_path=f"{REPO_ROOT}/eval")
)

app = modal.App("arewethesame-probe-seed31", image=image)
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
def probe(limit_items: int = 0):
    sys.path.insert(0, f"{REPO_ROOT}/scripts")
    import probe_adapters as pa

    output = Path("/outputs/probe_seed31/probe.json")
    args = argparse.Namespace(
        base_model=pa.DEFAULT_MODEL,
        adapters_root=Path("/outputs/lora_pilot_seed31"),
        items=Path(f"{REPO_ROOT}/eval/locked_v2_contextual/items.jsonl"),
        output=output,
        models=list(pa.DEFAULT_MODELS),
        limit_items=limit_items,
    )
    def checkpoint(partial: dict) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        out_vol.commit()

    try:
        report = pa.run(args, checkpoint=checkpoint)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        out_vol.commit()
        print("probe done", flush=True)
        return {"status": "ok", "items": report["items"], "models": list(report["models"])}
    except BaseException:
        tb = traceback.format_exc()
        print(tb, flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / "error.txt").write_text(tb, encoding="utf-8")
        out_vol.commit()
        raise


@app.local_entrypoint()
def main(limit_items: int = 0):
    result = probe.remote(limit_items=limit_items)
    print(result, flush=True)
