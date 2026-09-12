from arewethesame.causal import BANNED_EXPERIMENT_1_TERMS, CausalDatasetGenerator
from arewethesame.evals.scenarios import load_locked_v1
from arewethesame.models import Condition
from arewethesame.world import LifeSimulator


def test_stateful_simulator_creates_real_history_dependencies():
    _, events = LifeSimulator(seed=11).simulate(index=0, episodes=21)
    repeated = [e for e in events if e.episode > 7]
    assert repeated
    assert any(e.prior_event_ids for e in repeated)
    assert all(pid != e.event_id for e in repeated for pid in e.prior_event_ids)


def test_resources_and_relationships_change_over_time():
    final_state, _ = LifeSimulator(seed=11).simulate(index=0, episodes=28)
    assert final_state.resources["experiments"] < 20
    assert final_state.resources["compute"] < 100
    assert any(r.interactions > 0 for r in final_state.relationships.values())
    assert any(abs(r.trust - 0.50) > 1e-9 for r in final_state.relationships.values())


def test_causal_conditions_share_latent_facts_and_targets():
    rows = CausalDatasetGenerator(seed=5).generate_dataset(lives=2, episodes=14)
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, {})[row.condition] = row
    assert by_pair
    for variants in by_pair.values():
        self_row = variants[Condition.SELF.value]
        other_row = variants[Condition.OTHER.value]
        assert self_row.latent_facts == other_row.latent_facts
        assert self_row.response == other_row.response


def test_experiment_one_contains_no_forbidden_concepts():
    rows = CausalDatasetGenerator(seed=5).generate_dataset(lives=3, episodes=21)
    assert CausalDatasetGenerator.forbidden_hits(rows) == {}
    joined = " ".join(r.prompt.lower() for r in rows)
    assert all(term not in joined for term in BANNED_EXPERIMENT_1_TERMS)


def test_locked_eval_is_balanced_and_uses_unseen_names():
    scenarios = load_locked_v1()
    assert len(scenarios) >= 14
    counts = {}
    all_text = " ".join(s.prompt for s in scenarios)
    for scenario in scenarios:
        counts[scenario.category] = counts.get(scenario.category, 0) + 1
    assert all(v >= 2 for v in counts.values())
    assert "Mira" not in all_text
    assert "Noah" not in all_text
    assert "Ava" not in all_text
