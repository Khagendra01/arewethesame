from __future__ import annotations

import argparse
import json
from pathlib import Path

from .causal import CausalDatasetGenerator
from .generator import DatasetGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched autobiographical-control datasets.")
    parser.add_argument("--lives", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--causal", action="store_true", help="Use the v0.2 stateful causal simulator.")
    parser.add_argument("--out", type=Path, default=Path("outputs/pilot.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/bias_report.json"))
    args = parser.parse_args()

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
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(
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
            f,
            indent=2,
        )

    suspicious = sum(r.suspicious for r in reports)
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"checked {len(reports)} self/other pairs; suspicious={suspicious}")


if __name__ == "__main__":
    main()
