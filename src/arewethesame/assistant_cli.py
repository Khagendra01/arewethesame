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
    args = parser.parse_args(argv)

    builder = AssistantBatchBuilder(seed=args.seed)
    tasks = builder.prepare(
        lives=args.lives,
        episodes=args.episodes,
        variants=args.variants,
    )
    builder.write_jsonl(tasks, args.out)
    print(f"wrote {len(tasks)} assistant render tasks to {args.out}")
    print(
        "next: have ChatGPT fill one AssistantRender JSON object per pair_id, "
        "then run `arewethesame-assistant prepare-audit`"
    )


def _prepare_audit(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Create blinded X/Y audit tasks from assistant-curated canonical renders."
    )
    parser.add_argument("tasks", type=Path)
    parser.add_argument("renders", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/assistant_v03/audit_tasks.jsonl"),
    )
    args = parser.parse_args(argv)

    builder = AssistantBatchBuilder(seed=args.seed)
    tasks = builder.read_tasks(args.tasks)
    renders = builder.read_renders(args.renders)
    audit_tasks = builder.prepare_audits(tasks, renders)
    builder.write_jsonl(audit_tasks, args.out)
    print(f"wrote {len(audit_tasks)} blinded audit tasks to {args.out}")
    print(
        "next: have ChatGPT review only version_x/version_y + fact_catalog and "
        "write one AssistantAudit JSON object per pair_id"
    )


def _ingest(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest assistant renders + blind audits into five matched conditions."
    )
    parser.add_argument("tasks", type=Path)
    parser.add_argument("renders", type=Path)
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
    renders = builder.read_renders(args.renders)
    audits = builder.read_audits(args.audits)
    rows = builder.ingest(
        tasks,
        renders,
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
            "ChatGPT-in-the-loop v0.3 workflow. The simulator owns truth; "
            "assistant output is treated as auditable, untrusted language data."
        ),
    )
    parser.add_argument(
        "command",
        choices=("prepare", "prepare-audit", "ingest"),
    )
    args, rest = parser.parse_known_args()

    if args.command == "prepare":
        _prepare(rest)
    elif args.command == "prepare-audit":
        _prepare_audit(rest)
    else:
        _ingest(rest)


if __name__ == "__main__":
    main()
