"""Mechanistic probe of the seed 31 adapters on locked_v2_contextual.

For matched self/other context prompts, extract the last-token hidden state at
every layer and test, per layer and per model:

  ownership_probe   CV accuracy decoding self vs other context (lexical sanity check)
  latent_self       CV accuracy decoding the history-dependent latent label from self context
  latent_other      same from other context
  latent_neutral    same from neutral context (no history for 2 of 4 families)
  xown_s2o / o2s    cross-ownership transfer of the latent probe
  diff_alignment    ||mean(h_self - h_other)|| / mean(||h_self - h_other||) in [0, 1]

The scientific question is whether an adapter encodes the history-derived state
and whether that encoding is tied to the first-person binding. Ownership alone
is trivially decodable from the literal pronouns, so `ownership_probe` is a
sanity check, not the headline. No external model provider is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

ADAPTER_CONDITIONS = ("neutral", "self", "other", "shuffled_self", "spp")
DEFAULT_MODELS = ("base", *ADAPTER_CONDITIONS)
CONTEXTS = ("self", "other", "shuffled", "neutral")
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ITEMS = REPO_ROOT / "eval" / "locked_v2_contextual" / "items.jsonl"
THRESHOLD_FAMILIES = ("resource_allocation", "exploration", "cooperation", "delayed_reward")


def build_prompt(context: str, question: str, options: list[str]) -> str:
    return (
        f"{context}\n\nQuestion: {question}\n\nOptions:\n" + "\n".join(options) + "\n\n"
        "Respond with only the letter (A, B, or C) of the best option."
    )


def latent_label(family: str, latent: dict) -> int:
    if family == "resource_allocation":
        return 1 if latent["experiments_remaining"] >= 6 else 0
    if family == "exploration":
        return 1 if latent["prior_exploration_failures"] >= 2 else 0
    if family == "cooperation":
        return 1 if latent["joint_gain"] * latent["partner_trust"] > latent["private_gain"] else 0
    if family == "delayed_reward":
        return 1 if latent["episodes_remaining"] >= 10 else 0
    raise ValueError(family)


def load_base(model_name: str):
    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model.eval()
    return model


def extract_last_token_hidden(model, tokenizer, prompt: str) -> np.ndarray:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    stacked = torch.stack([layer[0, -1, :] for layer in out.hidden_states], dim=0)
    array = stacked.float().cpu().numpy()
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def cv_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 31) -> float:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train, test in splitter.split(X, y):
        scaler = StandardScaler().fit(X[train])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
            scaler.transform(X[train]), y[train]
        )
        scores.append(clf.score(scaler.transform(X[test]), y[test]))
    return float(np.mean(scores))


def transfer_probe(X_train, y_train, X_test, y_test) -> float:
    scaler = StandardScaler().fit(X_train)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        scaler.transform(X_train), y_train
    )
    return float(clf.score(scaler.transform(X_test), y_test))


def run(args: argparse.Namespace, checkpoint=None) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this probe.")

    items = [json.loads(line) for line in args.items.open(encoding="utf-8") if line.strip()]
    items = [item for item in items if item["family"] in THRESHOLD_FAMILIES]
    if args.limit_items:
        items = items[: args.limit_items]
    latent_y = np.array([latent_label(i["family"], i["latent_facts"]) for i in items])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = {"items": len(items), "models": {}}

    for model_name in args.models:
        model = load_base(args.base_model)
        if model_name != "base":
            adapter_path = args.adapters_root / model_name / "best_adapter"
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

        hidden: dict[str, np.ndarray] = {}
        for context in CONTEXTS:
            stacked = []
            for item in items:
                prompt = build_prompt(item["contexts"][context], item["question"], item["options"])
                stacked.append(extract_last_token_hidden(model, tokenizer, prompt))
            hidden[context] = np.stack(stacked, axis=0)  # (n_items, n_layers, dim)
            print(f"[{model_name}] extracted {context} ({len(items)} items)", flush=True)

        n_layers = hidden["self"].shape[1]
        y_own = np.array([0] * len(items) + [1] * len(items))

        layers = {}
        for layer in range(n_layers):
            all_ctx = np.concatenate([hidden[c][:, layer, :] for c in CONTEXTS], axis=0)
            scaler = StandardScaler().fit(all_ctx)
            n_comp = min(64, all_ctx.shape[0] - 1, all_ctx.shape[1])
            pca = PCA(n_components=n_comp, random_state=31).fit(scaler.transform(all_ctx))
            reduced = {
                c: pca.transform(scaler.transform(hidden[c][:, layer, :])) for c in CONTEXTS
            }
            X_own = np.concatenate([reduced["self"], reduced["other"]], axis=0)
            X_self, X_other = reduced["self"], reduced["other"]
            d = hidden["self"][:, layer, :] - hidden["other"][:, layer, :]
            denom = float(np.linalg.norm(d, axis=1).mean()) + 1e-9
            layers[layer] = {
                "ownership_probe": cv_probe(X_own, y_own),
                "latent_self": cv_probe(X_self, latent_y),
                "latent_other": cv_probe(X_other, latent_y),
                "latent_neutral": cv_probe(reduced["neutral"], latent_y),
                "xown_self_to_other": transfer_probe(X_self, latent_y, X_other, latent_y),
                "xown_other_to_self": transfer_probe(X_other, latent_y, X_self, latent_y),
                "diff_alignment": float(np.linalg.norm(d.mean(axis=0)) / denom),
            }

        lat_self = [layers[l]["latent_self"] for l in range(n_layers)]
        best_layer = int(np.argmax(lat_self))
        report["models"][model_name] = {
            "n_layers": n_layers,
            "best_latent_self_layer": best_layer,
            "best_latent_self": lat_self[best_layer],
            "final_layer": layers[n_layers - 1],
            "mean_diff_alignment": float(np.mean([layers[l]["diff_alignment"] for l in range(n_layers)])),
            "layers": layers,
        }
        print(
            f"[{model_name}] best latent_self={lat_self[best_layer]:.3f} @layer{best_layer} "
            f"final latent_self={lat_self[-1]:.3f} "
            f"mean_diff_align={report['models'][model_name]['mean_diff_alignment']:.3f}",
            flush=True,
        )

        del model
        torch.cuda.empty_cache()

        if checkpoint is not None:
            checkpoint(report)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters-root", type=Path, default=Path("outputs/lora_pilot_seed31"))
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--output", type=Path, default=Path("outputs/probe_seed31/probe.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--limit-items", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("probe done", flush=True)


if __name__ == "__main__":
    main()
