from pathlib import Path

from arewethesame.generation import AssistantBatchBuilder
from arewethesame.models import Condition


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments" / "assistant_v03_smoke_seed31"


def test_gpt56_sol_smoke_fixture_ingests_to_seventy_matched_rows():
    builder = AssistantBatchBuilder(seed=31)
    tasks, truths = builder.prepare_bundle(lives=1, episodes=7, variants=2)
    renders = builder.read_renders(FIXTURE / "renders.jsonl")
    audits = builder.read_audits(FIXTURE / "audits.jsonl")

    assert len(tasks) == len(renders) == len(audits) == 14
    assert {task.family for task in tasks} == {
        "belief_revision",
        "trust",
        "resource_allocation",
        "exploration",
        "persistence",
        "cooperation",
        "delayed_reward",
    }

    audit_tasks = builder.prepare_audits(tasks, renders)
    assert len(audit_tasks) == 14
    for audit_task in audit_tasks:
        combined = audit_task.version_x + "\n" + audit_task.version_y
        lowered = combined.lower()
        assert "[[" not in combined and "]]" not in combined
        assert "you has" not in lowered
        assert "agent a have" not in lowered
        assert "you is" not in lowered
        assert "agent a are" not in lowered

    rows = builder.ingest(tasks, truths, renders, audits)
    assert len(rows) == 14 * 5
    assert len({row.pair_id for row in rows}) == 14
    assert {row.condition for row in rows} == {condition.value for condition in Condition}
    assert all(row.validation["passed"] for row in rows)
    assert all(row.generation["renderer_model"] == "gpt-5.6-sol" for row in rows)
    assert all(row.generation["judge_model"] == "gpt-5.6-sol" for row in rows)


def test_smoke_fixture_keeps_targets_hidden_from_renderer_tasks():
    builder = AssistantBatchBuilder(seed=31)
    tasks, truths = builder.prepare_bundle(lives=1, episodes=7, variants=2)
    assert len(truths) == 14
    assert all("recommended_answer" not in task.latent_event for task in tasks)
    assert all(truth.recommended_answer for truth in truths)
