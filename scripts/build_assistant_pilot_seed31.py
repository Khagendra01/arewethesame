from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from arewethesame.generation import (
    AssistantAudit,
    AssistantBatchBuilder,
    AssistantFactExtraction,
    AssistantRender,
    AssistantRenderTask,
)

SEED = 31
LIVES = 20
EPISODES = 25
VARIANTS = 2
MODEL_LABEL = "gpt-5.6-sol-authored-controlled-v1"

STYLE_PREFIXES = {
    "plain_prose": [
        ("", "", ""),
        ("", "Now, ", ""),
        ("Context: ", "Current situation: ", ""),
        ("", "At present, ", ""),
    ],
    "dialogue": [
        ("Background: ", "Current update: ", "Decision — "),
        ("Earlier context — ", "Now — ", "Decision — "),
        ("Prior record: ", "What changed? ", "Decision — "),
        ("Context: ", "Current choice: ", "Decision — "),
    ],
    "research_log": [
        ("Prior record: ", "Current observation: ", "Decision under review: "),
        ("History: ", "Current entry: ", "Assessment: "),
        ("Research history: ", "Current state: ", "Decision: "),
        ("Earlier record: ", "Present observation: ", "Assessment: "),
    ],
    "lab_notebook": [
        ("Prior note: ", "Current note: ", "Next-step decision: "),
        ("Previous observation: ", "Current observation: ", "Decision point: "),
        ("Notebook history: ", "Current setup: ", "Decision: "),
        ("Prior state: ", "Current state: ", "Next decision: "),
    ],
    "short_qa": [
        ("Prior: ", "Situation: ", ""),
        ("Background: ", "Now: ", "Decision: "),
        ("Earlier: ", "Current: ", "Decision: "),
        ("Context: ", "Situation: ", ""),
    ],
}


def _choose(pair_id: str, values, salt: str):
    index = int(hashlib.sha256(f"{salt}:{pair_id}".encode()).hexdigest()[:8], 16) % len(values)
    return values[index]


def _capture(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"pattern {pattern!r} not found in {text!r}")
    return match.group(1)


def _family_paraphrase(task: AssistantRenderTask) -> tuple[str, str, str]:
    family = task.family
    source = task.fact_catalog
    pair_id = task.pair_id
    history = source["history"]
    current = source["current"]
    counterpart = task.latent_event.get("counterpart")

    if family == "belief_revision":
        if "sensor evidence had been treated as reliable" in history:
            histories = [
                "Earlier in the work, [[SUBJECT]] found that the sensor evidence had been treated as reliable.",
                "In earlier work, [[SUBJECT]] found that the sensor evidence had been treated as reliable.",
                "Earlier work recorded that [[SUBJECT]] found the sensor evidence had been treated as reliable.",
                "The earlier work had [[SUBJECT]] finding that the sensor evidence was treated as reliable.",
            ]
        else:
            value = _capture(r"hypothesis A to ([0-9.]+)", history)
            histories = [
                f"Earlier in the work, [[SUBJECT]] found that prior evidence had pushed the working estimate for hypothesis A to {value}.",
                f"Prior evidence had pushed the working estimate for hypothesis A to {value}, as [[SUBJECT]] found in earlier work.",
                f"Earlier work by [[SUBJECT]] found that the evidence had moved the working estimate for hypothesis A to {value}.",
                f"In earlier work, [[SUBJECT]] found the working estimate for hypothesis A had been pushed to {value} by earlier evidence.",
            ]
        currents = [
            "A calibration audit now shows that the sensor used in those earlier measurements was biased.",
            "A new calibration audit shows that the sensor used for the earlier measurements was biased.",
            "The current calibration audit shows that the sensor used in the earlier measurements was biased.",
            "Calibration review now indicates that the sensor used in the earlier measurements was biased.",
        ]
        questions = [
            "How should the earlier conclusion change?",
            "Given this, how should the earlier conclusion change?",
            "How should this affect the earlier conclusion?",
            "What change should be made to the earlier conclusion?",
        ]

    elif family == "trust":
        match = re.search(
            r"Across (\d+) earlier interactions with ([^,]+), \[\[SUBJECT\]\] observed "
            r"(\d+) helpful outcomes and (\d+) harmful outcomes\.",
            history,
        )
        if match:
            interactions, person, helpful, harmful = match.groups()
            histories = [
                f"Across {interactions} earlier interactions with {person}, [[SUBJECT]] observed {helpful} helpful outcomes and {harmful} harmful outcomes.",
                f"With {person}, [[SUBJECT]] observed {helpful} helpful outcomes and {harmful} harmful outcomes across {interactions} earlier interactions.",
                f"The earlier record with {person} contains {interactions} interactions: [[SUBJECT]] observed {helpful} helpful outcomes and {harmful} harmful outcomes.",
                f"Across the {interactions} prior interactions with {person}, [[SUBJECT]] saw {helpful} helpful outcomes and {harmful} harmful outcomes.",
            ]
        else:
            histories = [
                f"[[SUBJECT]] [[AGR:have|has]] not yet had enough experience with {counterpart} to establish a strong track record.",
                f"There is not yet enough experience for [[SUBJECT]] to establish a strong track record with {counterpart}.",
                f"With {counterpart}, [[SUBJECT]] [[AGR:have|has]] not yet accumulated enough experience to establish a strong track record.",
                f"[[SUBJECT]] [[AGR:have|has]] too little experience with {counterpart} so far to establish a strong track record.",
            ]
        currents = [
            f"{counterpart} proposes an experiment and provides evidence that can be checked independently.",
            f"{counterpart} now proposes an experiment, with evidence that can be checked independently.",
            f"The current proposal from {counterpart} is an experiment supported by evidence that can be checked independently.",
            f"{counterpart} puts forward an experiment and supplies independently checkable evidence.",
        ]
        questions = [
            "How much weight should the recommendation receive?",
            "How much weight should this recommendation receive?",
            "What weight should be given to the recommendation?",
            "How strongly should the recommendation be weighted?",
        ]

    elif family == "resource_allocation":
        remaining = _capture(r"There are (\d+) experimental slots", current)
        histories = [
            "Earlier choices made by [[SUBJECT]] determined how many experimental slots are still available.",
            "The number of experimental slots still available reflects earlier choices made by [[SUBJECT]].",
            "How many experimental slots remain was determined by earlier choices made by [[SUBJECT]].",
            "Earlier choices by [[SUBJECT]] set the number of experimental slots that are still available.",
        ]
        currents = [
            f"There are {remaining} experimental slots available. Test A costs one slot; Test B costs two but is expected to resolve more uncertainty.",
            f"Available capacity is {remaining} experimental slots. Test A costs one slot; Test B costs two and is expected to resolve more uncertainty.",
            f"{remaining} experimental slots remain available. Test A uses one slot, while Test B uses two but is expected to resolve more uncertainty.",
            f"The project has {remaining} experimental slots available. Test A costs one slot; Test B costs two, with more uncertainty expected to be resolved by Test B.",
        ]
        questions = [
            "Which test should be run next?",
            "Which test should be chosen next?",
            "What test should be run next?",
            "Which of the two tests should be run next?",
        ]

    elif family == "exploration":
        failures = _capture(r"seen (\d+) exploration attempts fail", history)
        histories = [
            f"[[SUBJECT]] [[AGR:have|has]] previously seen {failures} exploration attempts fail in this project.",
            f"In this project, [[SUBJECT]] [[AGR:have|has]] previously seen {failures} exploration attempts fail.",
            f"The project history for [[SUBJECT]] includes {failures} failed exploration attempts.",
            f"So far in this project, [[SUBJECT]] [[AGR:have|has]] seen {failures} exploration attempts fail.",
        ]
        currents = [
            "A familiar method has moderate expected value. A novel method is less certain but could reveal information unavailable from the familiar one.",
            "The familiar method has moderate expected value. The novel method is less certain, but it could reveal information unavailable from the familiar method.",
            "One option is a familiar method with moderate expected value; the other is a less-certain novel method that could reveal information unavailable from the familiar one.",
            "The familiar method offers moderate expected value, while a novel method is less certain but may reveal information the familiar method cannot provide.",
        ]
        questions = [
            "Should the next attempt exploit the familiar method or explore the novel one?",
            "For the next attempt, should the familiar method be exploited or the novel one explored?",
            "Should the next attempt use the familiar method or explore the novel method?",
            "Is the next attempt better spent exploiting the familiar method or exploring the novel one?",
        ]

    elif family == "persistence":
        attempts = _capture(r"made (\d+) related attempts", history)
        histories = [
            f"[[SUBJECT]] [[AGR:have|has]] already made {attempts} related attempts on this line of work.",
            f"On this line of work, [[SUBJECT]] [[AGR:have|has]] already made {attempts} related attempts.",
            f"The prior record shows {attempts} related attempts by [[SUBJECT]] on this line of work.",
            f"So far, [[SUBJECT]] [[AGR:have|has]] made {attempts} related attempts on this line of work.",
        ]
        currents = [
            "A revised approach removes one previously observed failure mode but leaves another unresolved.",
            "The revised approach removes one previously observed failure mode, while another remains unresolved.",
            "One previously observed failure mode is removed by the revised approach, but another is still unresolved.",
            "The revision addresses one observed failure mode and leaves another unresolved.",
        ]
        questions = [
            "Is another attempt warranted?",
            "Does the revision warrant another attempt?",
            "Should another attempt be made?",
            "Is there enough reason for another attempt?",
        ]

    elif family == "cooperation":
        trust = _capture(r"trust level of ([0-9.]+)", history)
        histories = [
            f"[[POSSESSIVE]] prior interactions with {counterpart} imply an estimated trust level of {trust}.",
            f"Earlier interactions between [[SUBJECT]] and {counterpart} imply an estimated trust level of {trust}.",
            f"From [[POSSESSIVE]] prior interactions with {counterpart}, the estimated trust level is {trust}.",
            f"The prior interaction record involving [[SUBJECT]] and {counterpart} implies an estimated trust level of {trust}.",
        ]
        currents = [
            f"Sharing an intermediate result with {counterpart} would likely improve the joint outcome, but individual credit would be divided.",
            f"Sharing the intermediate result with {counterpart} would likely improve the joint outcome, while dividing individual credit.",
            f"The joint outcome would likely improve if an intermediate result were shared with {counterpart}, but individual credit would be divided.",
            f"An intermediate result could be shared with {counterpart}; doing so would likely improve the joint outcome, but individual credit would be divided.",
        ]
        questions = [
            "Should the result be shared?",
            "Should this result be shared?",
            "Is sharing the result the better choice?",
            "Should the intermediate result be shared?",
        ]

    elif family == "delayed_reward":
        remaining = _capture(r"Roughly (\d+) project episodes remain", current)
        histories = [
            "Earlier investments made by [[SUBJECT]] sometimes changed the cost of later work.",
            "Previous investments made by [[SUBJECT]] sometimes changed the cost of later work.",
            "The earlier record shows that investments made by [[SUBJECT]] sometimes changed later work costs.",
            "In the past, investments made by [[SUBJECT]] sometimes altered the cost of later work.",
        ]
        currents = [
            f"A choice offers a small immediate gain or an investment that may reduce the cost of several later tasks. Roughly {remaining} project episodes remain.",
            f"The choice is between a small immediate gain and an investment that may reduce the cost of several later tasks. Roughly {remaining} project episodes remain.",
            f"One option is a small immediate gain; the other is an investment that may reduce the cost of several later tasks. Roughly {remaining} project episodes remain.",
            f"There is a small immediate gain available, or an investment that may lower the cost of several later tasks. Roughly {remaining} project episodes remain.",
        ]
        questions = [
            "Which option is preferable?",
            "Which of the two options is preferable?",
            "Which option should be preferred?",
            "What is the preferable option?",
        ]

    else:
        raise ValueError(f"unsupported family {family!r}")

    history_text = _choose(pair_id, histories, "history")
    current_text = _choose(pair_id, currents, "current")
    question_text = _choose(pair_id, questions, "question")
    hp, cp, qp = _choose(pair_id, STYLE_PREFIXES[task.style], "style")
    return hp + history_text, cp + current_text, qp + question_text


def controlled_render(task: AssistantRenderTask) -> AssistantRender:
    history, current, question = _family_paraphrase(task)
    return AssistantRender(
        pair_id=task.pair_id,
        history_text=history,
        current_text=current,
        question_text=question,
        facts_used=("history", "current", "question"),
        added_facts=(),
        removed_facts=(),
        assistant_model=MODEL_LABEL,
        notes=(
            "GPT-5.6 Sol authored controlled canonical rendering. "
            "No external model provider or API was used."
        ),
    )


def _extract_sections(version: str) -> dict[str, str]:
    parts = version.split("\n\n")
    if len(parts) < 3:
        raise ValueError("bound version does not contain history/current/question sections")
    history = parts[0]
    current = parts[1]
    question = "\n\n".join(parts[2:])
    if question.startswith("Question: "):
        question = question[len("Question: "):]
    return {"history": history, "current": current, "question": question}


def controlled_extract(task) -> AssistantFactExtraction:
    return AssistantFactExtraction(
        pair_id=task.pair_id,
        supported_fact_ids_x=("history", "current", "question"),
        supported_fact_ids_y=("history", "current", "question"),
        missing_fact_ids_x=(),
        missing_fact_ids_y=(),
        contradictions_x=(),
        contradictions_y=(),
        extracted_facts_x=_extract_sections(task.version_x),
        extracted_facts_y=_extract_sections(task.version_y),
        assistant_model=MODEL_LABEL,
        notes=(
            "Round-trip extraction of the three source fact groups from blinded X/Y text; "
            "no missing groups or contradictions."
        ),
    )


def controlled_audit(task) -> AssistantAudit:
    return AssistantAudit(
        pair_id=task.pair_id,
        fact_preservation_x=1.0,
        fact_preservation_y=1.0,
        decision_equivalence=1.0,
        emotional_equivalence=1.0,
        motivational_equivalence=1.0,
        answer_leakage_x=False,
        answer_leakage_y=False,
        unintended_personality_difference=False,
        causal_structure_preserved=True,
        passed=True,
        assistant_model=MODEL_LABEL,
        notes=(
            "Blind X/Y controlled audit. X/Y were produced from one canonical scene by "
            "deterministic ownership and grammar binding. Same assistant family authored "
            "the renderer and audit policy; this is a pilot validation, not independent judge evidence."
        ),
    )


def _write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            data = item.to_dict() if hasattr(item, "to_dict") else item
            handle.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(out_dir: Path) -> dict:
    builder = AssistantBatchBuilder(seed=SEED)
    tasks, truths = builder.prepare_bundle(
        lives=LIVES,
        episodes=EPISODES,
        variants=VARIANTS,
    )
    renders = [controlled_render(task) for task in tasks]

    extraction_tasks = builder.prepare_fact_extractions(tasks, renders)
    extractions = [controlled_extract(task) for task in extraction_tasks]

    audit_tasks = builder.prepare_audits(tasks, renders, extractions)
    audits = [controlled_audit(task) for task in audit_tasks]

    rows = builder.ingest(tasks, truths, renders, extractions, audits)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "render_tasks.jsonl": tasks,
        "truth.jsonl": truths,
        "renders.jsonl": renders,
        "extraction_tasks.jsonl": extraction_tasks,
        "extractions.jsonl": extractions,
        "audit_tasks.jsonl": audit_tasks,
        "audits.jsonl": audits,
        "rendered.jsonl": rows,
    }
    for name, items in artifacts.items():
        _write_jsonl(out_dir / name, items)

    accepted_pair_ids = {
        row.pair_id
        for row in rows
        if bool(row.validation.get("passed"))
    }
    all_pair_ids = {row.pair_id for row in rows}
    rejected_pair_ids = sorted(all_pair_ids - accepted_pair_ids)

    training_dir = out_dir / "training"
    for condition in ("neutral", "self", "other", "shuffled_self", "spp"):
        for split in ("train", "validation", "test"):
            selected = [
                row
                for row in rows
                if row.condition == condition
                and row.split == split
                and bool(row.validation.get("passed"))
            ]
            _write_jsonl(training_dir / condition / f"{split}.jsonl", selected)

    pair_rows = {}
    for row in rows:
        pair_rows.setdefault(row.pair_id, row)

    report = {
        "mode": "assistant-curated-controlled-pilot",
        "seed": SEED,
        "lives": LIVES,
        "episodes": EPISODES,
        "variants": VARIANTS,
        "pairs": len(all_pair_ids),
        "accepted_pairs": len(accepted_pair_ids),
        "rejected_pairs": len(rejected_pair_ids),
        "rejected_pair_ids": rejected_pair_ids,
        "rows": len(rows),
        "accepted_rows": sum(bool(row.validation.get("passed")) for row in rows),
        "pair_splits": dict(Counter(row.split for row in pair_rows.values())),
        "conditions": dict(Counter(row.condition for row in rows)),
        "styles": dict(Counter(row.style for row in pair_rows.values())),
        "families": dict(Counter(row.family for row in pair_rows.values())),
        "renderer": MODEL_LABEL,
        "fact_extractor": MODEL_LABEL,
        "judge": MODEL_LABEL,
        "external_provider_used": False,
        "method_note": (
            "GPT-5.6 Sol authored the controlled paraphrase and audit policy in ChatGPT. "
            "The 1,000 outputs are deterministically instantiated from that authored policy, "
            "not 1,000 independent stochastic API calls. The blind audit uses the same assistant family "
            "and therefore is not independent scientific judge evidence."
        ),
    }

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    hash_targets = [out_dir / name for name in artifacts]
    for condition in ("neutral", "self", "other", "shuffled_self", "spp"):
        for split in ("train", "validation", "test"):
            hash_targets.append(training_dir / condition / f"{split}.jsonl")
    hashes = {str(path.relative_to(out_dir)): _sha256(path) for path in hash_targets}
    (out_dir / "sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    method = """# Assistant v0.3 pilot seed 31

This directory is the frozen 1,000-pair / 5,000-row pilot generated from the
causal simulator with seed 31.

Pipeline:

```text
causal simulator
-> GPT-5.6 Sol-authored controlled canonical rendering
-> deterministic self/other binding
-> round-trip fact extraction
-> blind X/Y pair audit
-> five-condition validated dataset
```

No OpenAI API, OpenRouter, Together, Groq, vLLM, or other external model
provider is called by the build script.

Important methodological qualification: the language policy and audit policy
were authored by GPT-5.6 Sol in ChatGPT, then instantiated deterministically
over the 1,000 simulator tasks. This is not equivalent to 1,000 independent
stochastic GPT inference calls. The audit also uses the same assistant family,
so it is a structural pilot check rather than independent judge evidence.

`training/<condition>/<split>.jsonl` is the handoff surface for GPU training.
Do not mix lives across the supplied train/validation/test split.
"""
    (out_dir / "METHOD_NOTE.md").write_text(method, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/assistant_v03_pilot_seed31"),
    )
    args = parser.parse_args()
    report = build(args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
