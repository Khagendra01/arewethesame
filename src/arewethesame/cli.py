from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .causal import CausalDatasetGenerator
from .generation import NaturalizedDatasetBuilder
from .generator import DatasetGenerator
from .providers import make_text_model


def _legacy(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Generate matched autobiographical-control datasets.")
    parser.add_argument("--lives", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--causal", action="store_true", help="Use the v0.2 stateful causal simulator.")
    parser.add_argument("--out", type=Path, default=Path("outputs/pilot.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/bias_report.json"))
    args = parser.parse_args(argv)

    if args.causal:
        gen = CausalDatasetGenerator(seed=args.seed)
        rows = gen.generate_dataset(lives=args.lives, episodes=args.episodes)
        gen.write_jsonl(rows, args.out)
        report = {
            "mode": "causal-v0.2",
            "rows": len(rows),
            "pairs": len({r.pair_id for r in rows}),
            "forbidden_hits": gen.forbidden_hits(rows),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {len(rows)} causal rows to {args.out}")
        print(f"forbidden concept hits: {report['forbidden_hits']}")
        return

    gen = DatasetGenerator(seed=args.seed)
    rows = gen.generate_dataset(lives=args.lives, episodes=args.episodes)
    gen.write_jsonl(rows, args.out)
    reports = gen.bias_reports(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            [
                {
                    "pair_id": r.pair_id,
                    "lexical_similarity": r.lexical_similarity,
                    "length_ratio": r.length_ratio,
                    "emotion_delta": r.emotion_delta,
                    "motivation_delta": r.motivation_delta,
                    "suspicious": r.suspicious,
                }
                for r in reports
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    suspicious = sum(r.suspicious for r in reports)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"checked {len(reports)} self/other pairs; suspicious={suspicious}")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="v0.3 natural-language rendering + blind validation pipeline.")
    parser.add_argument("--lives", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=14)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--provider", choices=("deterministic", "openai-compatible"), default="deterministic")
    parser.add_argument("--model")
    parser.add_argument("--judge-provider", choices=("deterministic", "openai-compatible"))
    parser.add_argument("--judge-model")
    parser.add_argument("--base-url")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--out", type=Path, default=Path("outputs/rendered_v03.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/rendered_v03_report.json"))
    args = parser.parse_args(argv)

    renderer = make_text_model(args.provider, model=args.model, base_url=args.base_url, api_key=args.api_key)
    judge_provider = args.judge_provider or args.provider
    judge_model = args.judge_model or args.model
    judge = make_text_model(
        judge_provider,
        model=judge_model,
        base_url=args.judge_base_url or args.base_url,
        api_key=args.judge_api_key or args.api_key,
    )
    builder = NaturalizedDatasetBuilder(renderer, judge, seed=args.seed)
    rows = builder.generate(lives=args.lives, episodes=args.episodes, variants=args.variants)
    builder.write_jsonl(rows, args.out)

    unique_pairs = {}
    for row in rows:
        unique_pairs.setdefault(row.pair_id, bool(row.validation.get("passed")))
    condition_counts = Counter(row.condition for row in rows)
    split_counts = Counter(row.split for row in rows)
    report = {
        "mode": "naturalized-v0.3",
        "rows": len(rows),
        "pairs": len(unique_pairs),
        "accepted_pairs": sum(unique_pairs.values()),
        "rejected_pairs": len(unique_pairs) - sum(unique_pairs.values()),
        "acceptance_rate": sum(unique_pairs.values()) / max(1, len(unique_pairs)),
        "conditions": dict(condition_counts),
        "splits": dict(split_counts),
        "renderer_model": renderer.model_name,
        "judge_model": judge.model_name,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows / {len(unique_pairs)} pairs to {args.out}")
    print(f"accepted pairs: {report['accepted_pairs']}/{report['pairs']} ({report['acceptance_rate']:.1%})")


def _validate(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Summarize stored v0.3 validation results.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, default=Path("outputs/validation_summary.json"))
    args = parser.parse_args(argv)
    rows = _read_jsonl(args.input)
    by_pair: dict[str, bool] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], bool(row.get("validation", {}).get("passed")))
    failures = [pair_id for pair_id, passed in by_pair.items() if not passed]
    report = {
        "input": str(args.input),
        "rows": len(rows),
        "pairs": len(by_pair),
        "accepted_pairs": sum(by_pair.values()),
        "rejected_pairs": len(failures),
        "failed_pair_ids": failures[:100],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _build_dataset(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Filter validated v0.3 renderings into a training-ready JSONL.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--condition", choices=("neutral", "self", "other", "shuffled_self", "spp", "all"), default="all")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/training_ready.jsonl"))
    args = parser.parse_args(argv)
    rows = _read_jsonl(args.input)
    selected = []
    for row in rows:
        if not args.include_rejected and not row.get("validation", {}).get("passed", False):
            continue
        if args.condition != "all" and row.get("condition") != args.condition:
            continue
        if args.split != "all" and row.get("split") != args.split:
            continue
        selected.append(row)
    _write_jsonl(selected, args.out)
    print(f"wrote {len(selected)} validated rows to {args.out}")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"render", "validate", "build-dataset"}:
        command, rest = argv[0], argv[1:]
        if command == "render":
            _render(rest)
        elif command == "validate":
            _validate(rest)
        else:
            _build_dataset(rest)
        return
    _legacy(argv)


if __name__ == "__main__":
    main()
