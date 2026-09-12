import re

import pytest

from arewethesame.generation import (
    AssistantAudit,
    AssistantBatchBuilder,
    AssistantRender,
)
from arewethesame.models import Condition


def _placeholder_history(text: str) -> str:
    text = re.sub(r"\bYour\b", "[[POSSESSIVE]]", text)
    text = re.sub(r"\byour\b", "[[POSSESSIVE]]", text)
    text = re.sub(r"\bYou\b", "[[SUBJECT]]", text)
    return re.sub(r"\byou\b", "[[SUBJECT]]", text)


def _render_for(task) -> AssistantRender:
    return AssistantRender(
        pair_id=task.pair_id,
        history_text=_placeholder_history(task.fact_catalog["history"]),
        current_text=task.fact_catalog["current"],
        question_text=task.fact_catalog["question"],
        facts_used=("history", "current", "question"),
        added_facts=(),
        removed_facts=(),
        assistant_model="gpt-5.6-sol-test",
    )


def _audit_for(pair_id: str) -> AssistantAudit:
    return AssistantAudit(
        pair_id=pair_id,
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
        assistant_model="gpt-5.6-sol-test",
    )


def test_assistant_prepare_emits_one_task_per_scene_variant_without_target_leakage():
    builder = AssistantBatchBuilder(seed=31)
    tasks, truths = builder.prepare_bundle(lives=2, episodes=7, variants=2)
    assert len(tasks) == 2 * 7 * 2
    assert len(truths) == len(tasks)
    assert len({task.pair_id for task in tasks}) == len(tasks)
    assert all(task.latent_event["event_id"] == task.event_id for task in tasks)
    assert all("recommended_answer" not in task.latent_event for task in tasks)
    assert all(set(task.fact_catalog) == {"history", "current", "question"} for task in tasks)

    truth_by_pair = {truth.pair_id: truth for truth in truths}
    assert all(truth_by_pair[task.pair_id].recommended_answer for task in tasks)

    by_pair = {task.pair_id: task for task in tasks}
    shuffled = [task for task in tasks if task.shuffled_pair_id != task.pair_id]
    assert shuffled
    for task in shuffled:
        assert by_pair[task.shuffled_pair_id].style == task.style
        assert by_pair[task.shuffled_pair_id].family != task.family


def test_assistant_audit_tasks_hide_condition_identity_and_use_source_truth():
    builder = AssistantBatchBuilder(seed=31)
    tasks = builder.prepare(lives=1, episodes=7, variants=1)
    renders = [_render_for(task) for task in tasks]
    audit_tasks = builder.prepare_audits(tasks, renders)
    assert len(audit_tasks) == len(tasks)
    task_by_pair = {task.pair_id: task for task in tasks}

    for audit_task in audit_tasks:
        payload = audit_task.to_dict()
        assert "blinded_order" not in payload
        assert "condition" not in payload
        assert audit_task.version_x != audit_task.version_y
        combined = audit_task.version_x + audit_task.version_y
        assert "Agent A" in combined
        assert "you" in combined.lower() or "your" in combined.lower()

        neutral_history = audit_task.fact_catalog["history"]
        assert "[[SUBJECT]]" in neutral_history or "[[POSSESSIVE]]" in neutral_history
        assert "Agent A" not in neutral_history
        assert not re.search(r"\byou\b|\byour\b", neutral_history, flags=re.IGNORECASE)
        source = task_by_pair[audit_task.pair_id].fact_catalog
        assert audit_task.fact_catalog["current"] == source["current"]
        assert audit_task.fact_catalog["question"] == source["question"]


def test_assistant_render_rejects_numeric_fact_drift():
    builder = AssistantBatchBuilder(seed=31)
    tasks = builder.prepare(lives=1, episodes=7, variants=1)
    task = next(task for task in tasks if re.search(r"\d", task.fact_catalog["current"]))
    render = _render_for(task)
    bad = AssistantRender(
        pair_id=render.pair_id,
        history_text=render.history_text,
        current_text=re.sub(r"\d+(?:\.\d+)?", "999", render.current_text, count=1),
        question_text=render.question_text,
        facts_used=render.facts_used,
        added_facts=(),
        removed_facts=(),
        assistant_model=render.assistant_model,
    )
    with pytest.raises(ValueError, match="numeric facts changed"):
        builder.prepare_audits([task], [bad])


def test_assistant_ingest_builds_five_conditions_without_text_model():
    builder = AssistantBatchBuilder(seed=31)
    tasks, truths = builder.prepare_bundle(lives=2, episodes=7, variants=2)
    renders = [_render_for(task) for task in tasks]
    audits = [_audit_for(task.pair_id) for task in tasks]
    rows = builder.ingest(tasks, truths, renders, audits)

    assert len(rows) == len(tasks) * 5
    assert {row.condition for row in rows} == {condition.value for condition in Condition}
    assert all(row.generation["mode"] == "assistant-curated" for row in rows)
    assert all(row.validation["passed"] for row in rows)
