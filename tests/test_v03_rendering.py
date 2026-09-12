from arewethesame.generation import NaturalizedDatasetBuilder, split_for_life
from arewethesame.models import Condition
from arewethesame.providers import DeterministicTextModel
from arewethesame.rendering import CanonicalRenderer, PerspectiveRenderer
from arewethesame.validation import ValidationPipeline
from arewethesame.world import LifeSimulator


def test_canonical_render_uses_protected_subject_placeholders():
    _, events = LifeSimulator(seed=31).simulate(0, 7)
    scene = CanonicalRenderer(DeterministicTextModel()).render(events[0], style="plain_prose")
    assert "[[SUBJECT]]" in scene.history_text or "[[POSSESSIVE]]" in scene.history_text
    assert scene.added_facts == ()
    assert scene.removed_facts == ()


def test_self_other_are_same_scene_with_only_ownership_changed():
    _, events = LifeSimulator(seed=31).simulate(0, 7)
    scene = CanonicalRenderer(DeterministicTextModel()).render(events[0], style="research_log")
    renderer = PerspectiveRenderer()
    self_text = renderer.render(scene, Condition.SELF)
    other_text = renderer.render(scene, Condition.OTHER)
    assert "you" in self_text.lower() or "your" in self_text.lower()
    assert "Agent A" in other_text
    assert scene.current_text in self_text and scene.current_text in other_text
    assert scene.question_text in self_text and scene.question_text in other_text


def test_agreement_slots_keep_both_perspectives_grammatical():
    _, events = LifeSimulator(seed=31).simulate(0, 7)
    scene = CanonicalRenderer(DeterministicTextModel()).render(events[1], style="plain_prose")
    assert "[[AGR:have|has]]" in scene.history_text

    renderer = PerspectiveRenderer()
    self_text = renderer.render(scene, Condition.SELF)
    other_text = renderer.render(scene, Condition.OTHER)
    assert "you have not yet" in self_text
    assert "Agent A has not yet" in other_text
    assert "you has" not in self_text
    assert "Agent A have" not in other_text
    assert "[[AGR:" not in self_text + other_text


def test_round_trip_validation_passes_for_controlled_pair():
    _, events = LifeSimulator(seed=31).simulate(0, 7)
    model = DeterministicTextModel()
    scene = CanonicalRenderer(model).render(events[1], style="plain_prose")
    renderer = PerspectiveRenderer()
    result = ValidationPipeline(model).validate(
        "pair-test",
        renderer.render(scene, Condition.SELF),
        renderer.render(scene, Condition.OTHER),
        scene.fact_catalog,
    )
    assert result.deterministic.passed
    assert result.semantic_similarity >= 0.9
    assert result.self_facts.score >= 0.70
    assert result.other_facts.score >= 0.70
    assert result.judge.passed
    assert result.passed


def test_naturalized_builder_generates_five_conditions_per_variant():
    rows = NaturalizedDatasetBuilder(DeterministicTextModel(), seed=31).generate(
        lives=2, episodes=7, variants=2
    )
    assert len(rows) == 2 * 7 * 2 * 5
    assert all(row.validation["passed"] for row in rows)
    assert {row.condition for row in rows} == {c.value for c in Condition}


def test_life_level_split_never_varies_with_episode_or_condition():
    ids = [f"life_{i:04d}" for i in range(500)]
    mapping = {life_id: split_for_life(life_id) for life_id in ids}
    assert set(mapping.values()) == {"train", "validation", "test"}
    for life_id, split in mapping.items():
        assert split_for_life(life_id) == split
