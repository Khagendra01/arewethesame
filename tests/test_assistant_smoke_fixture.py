from pathlib import Path

from arewethesame.generation import AssistantBatchBuilder
from arewethesame.models import Condition


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments" / "assistant_v03_smoke_seed31"


def test_gpt56_sol_smoke_fixture_ingests_to_seventy_matched_rows():
    builder = AssistantBatchBuilder(seed=31)
    tasks, truths = builder.prepare_bundle(lives=1, episodes=7, variants=2)
    renders = builder.read_renders(FIXTURE / "renders.jsonl")
    extractions = builder.read_extractions(FIXTURE / "extractions.jsonl")
    audits = builder.read_audits(FIXTURE / "audits.jsonl")

    assert len(tasks) == len(renders) == len(extractions) == len(audits) == 14
    assert {task.family for task in tasks} == {
        "belief_revision",
        "trust",
        "resource_allocation",
        "exploration",
        "persistence",
        "cooperation",
        "delayed_reward",
    }

    extraction_tasks = builder.prepare_fact_extractions(tasks, renders)
    assert len(extraction_tasks) == 14
    for extraction_task in extraction_tasks:
        combined = extraction_task.version_x + "\n" + extraction_task.version_y
        lowered = combined.lower()
        assert "[[" not in combined and "]]" not in combined
        assert "you has" not in lowered
        assert "agent a have" not in lowered
        assert "you is" not in lowered
        assert "agent a are" not in lowered

    audit_tasks = builder.prepare_audits(tasks, renders, extractions)
    assert len(audit_tasks) == 14

    rows = builder.ingest(tasks, truths, renders, extractions, audits)
    assert len(rows) == 14 * 5
    assert len({row.pair_id for row in rows}) == 14
    assert {row.condition for row in rows} == {condition.value for condition in Condition}
    assert all(row.validation["round_trip_fact_extraction"]["passed"] for row in rows)
    assert all(row.validation["passed"] for row in rows)
    assert all(row.generation["renderer_model"] == "gpt-5.6-sol" for row in rows)
    assert all(row.generation["fact_extractor_model"] == "gpt-5.6-sol" for row in rows)
    assert all(row.generation["judge_model"] == "gpt-5.6-sol" for row in rows)


def test_smoke_fixture_keeps_targets_hidden_from_all_inference_passes():
    builder = AssistantBatchBuilder(seed=31)
    tasks, truths = builder.prepare_bundle(lives=1, episodes=7, variants=2)
    assert len(truths) == 14
    assert all("recommended_answer" not in task.latent_event for task in tasks)
    assert all(truth.recommended_answer for truth in truths)

    renders = builder.read_renders(FIXTURE / "renders.jsonl")
    extraction_tasks = builder.prepare_fact_extractions(tasks, renders)
    payload_text = "\n".join(str(task.to_dict()) for task in extraction_tasks)
    assert all(truth.recommended_answer not in payload_text for truth in truths)
