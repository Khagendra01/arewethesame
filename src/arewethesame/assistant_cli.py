from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .generation import AssistantBatchBuilder


def _prepare(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare simulator-owned render tasks for ChatGPT curation."
    )
    parser.add_argument("--lives", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/assistant_v03/render_tasks.jsonl"),
    )
    parser.add_argument(
        "--truth-out",
        type=Path,
        default=Path("outputs/assistant_v03/truth.jsonl"),
        help="Simulator targets for ingest only; do not provide this file to renderer, extractor, or blind auditor.",
    )
    args = parser.parse_args(argv)

    builder = AssistantBatchBuilder(seed=args.seed)
    tasks, truths = builder.prepare_bundle(
        lives=args.lives,
        episodes=args.episodes,
        variants=args.variants,
    )
    builder.write_jsonl(tasks, args.out)
    builder.write_jsonl(truths, args.truth_out)
    print(f"wrote {len(tasks)} assistant render tasks to {args.out}")
    print(f"wrote {len(truths)} simulator truth records to {args.truth_out}")
    print("do not expose the truth file during rendering, extraction, or blind auditing")
    print(
        "next: have ChatGPT write one AssistantRender per pair_id, then run "
        "`arewethesame-assistant prepare-extraction`"
    )


def _prepare_extraction(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Create blind X/Y round-trip fact extraction tasks after deterministic perspective binding."
    )
    parser.add_argument("tasks", type=Path)
    parser.add_argument("renders", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/assistant_v03/extraction_tasks.jsonl"),
    )
    args = parser.parse_args(argv)

    builder = AssistantBatchBuilder(seed=args.seed)
    tasks = builder.read_tasks(args.tasks)
    renders = builder.read_renders(args.renders)
    extraction_tasks = builder.prepare_fact_extractions(tasks, renders)
    builder.write_jsonl(extraction_tasks, args.out)
    print(f"wrote {len(extraction_tasks)} fact extraction tasks to {args.out}")
    print(
        "next: have ChatGPT extract the facts from Version X and Version Y into "
        "one AssistantFactExtraction object per pair_id"
    )


def _prepare_audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Create blinded X/Y judge tasks only for pairs that pass round-trip fact extraction."
    )
    parser.add_argument("tasks", type=Path)
    parser.add_argument("renders", type=Path)
    parser.add_argument("extractions", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--min-fact-score", type=float, default=0.95)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/assistant_v03/audit_tasks.jsonl"),
    )
    args = parser.parse_args(argv)

    builder = AssistantBatchBuilder(seed=args.seed)
    tasks = builder.read_tasks(args.tasks)
    renders = builder.read_renders(args.renders)
    extractions = builder.read_extractions(args.extractions)
    audit_tasks = builder.prepare_audits(
        tasks,
        renders,
        extractions,
        min_fact_score=args.min_fact_score,
    )
    builder.write_jsonl(audit_tasks, args.out)
    print(f"wrote {len(audit_tasks)} blinded judge tasks to {args.out}")
    print(
        "next: have ChatGPT judge only version_x/version_y + source fact_catalog and "
        "write one AssistantAudit object for each emitted pair_id"
    )


def _ingest(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest simulator truth + assistant renders + round-trip extractions + "
            "blind audits into five matched conditions."
        )
    )
    parser.add_argument("tasks", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("renders", type=Path)
    parser.add_argument("extractions", type=Path)
    parser.add_argument("audits", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--min-fact-score", type=float, default=0.95)
    parser.add_argument("--min-semantic", type=float, default=0.90)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/assistant_v03/rendered.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/assistant_v03/report.json"),
    )
    args = parser.parse_args(argv)

    builder = AssistantBatchBuilder(seed=args.seed)
    tasks = builder.read_tasks(args.tasks)
    truths = builder.read_truths(args.truth)
    renders = builder.read_renders(args.renders)
    extractions = builder.read_extractions(args.extractions)
    audits = builder.read_audits(args.audits)
    rows = builder.ingest(
        tasks,
        truths,
        renders,
        extractions,
        audits,
        min_fact_score=args.min_fact_score,
        min_semantic=args.min_semantic,
    )
    builder.write_jsonl(rows, args.out)

    pair_pass: dict[str, bool] = {}
    for row in rows:
        pair_pass.setdefault(row.pair_id, bool(row.validation.get("passed")))
    report = {
        "mode": "assistant-curated-v0.3",
        "rows": len(rows),
        "pairs": len(pair_pass),
        "accepted_pairs": sum(pair_pass.values()),
        "rejected_pairs": len(pair_pass) - sum(pair_pass.values()),
        "acceptance_rate": sum(pair_pass.values()) / max(1, len(pair_pass)),
        "conditions": dict(Counter(row.condition for row in rows)),
        "splits": dict(Counter(row.split for row in rows)),
        "styles": dict(Counter(row.style for row in rows)),
        "families": dict(Counter(row.family for row in rows)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows / {len(pair_pass)} pairs to {args.out}")
    print(
        f"accepted pairs: {report['accepted_pairs']}/{report['pairs']} "
        f"({report['acceptance_rate']:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arewethesame-assistant",
        description=(
            "No-API ChatGPT-in-the-loop v0.3 workflow: simulator -> ChatGPT render -> "
            "deterministic binding -> ChatGPT fact extraction -> ChatGPT blind judge -> ingest."
        ),
    )
    parser.add_argument(
        "command",
        choices=("prepare", "prepare-extraction", "prepare-audit", "ingest"),
    )
    args, rest = parser.parse_known_args()

    if args.command == "prepare":
        _prepare(rest)
    elif args.command == "prepare-extraction":
        _prepare_extraction(rest)
    elif args.command == "prepare-audit":
        _prepare_audit(rest)
    else:
        _ingest(rest)


if __name__ == "__main__":
    main()
