from arewethesame import Condition, DatasetGenerator


def test_each_event_has_all_conditions():
    rows = DatasetGenerator(seed=1).generate_dataset(lives=2, episodes=6)
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, set()).add(row.condition)
    assert by_pair
    assert all(conditions == set(Condition) for conditions in by_pair.values())


def test_self_and_other_share_source_facts_and_target():
    rows = DatasetGenerator(seed=2).generate_dataset(lives=2, episodes=6)
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row.pair_id, {})[row.condition] = row
    for variants in by_pair.values():
        self_row = variants[Condition.SELF]
        other_row = variants[Condition.OTHER]
        assert self_row.source_facts == other_row.source_facts
        assert self_row.response == other_row.response


def test_no_emotion_or_mortality_leakage_in_matched_pairs():
    gen = DatasetGenerator(seed=3)
    rows = gen.generate_dataset(lives=3, episodes=12)
    reports = gen.bias_reports(rows)
    assert reports
    assert all(r.emotion_delta == 0 for r in reports)
    assert all(r.motivation_delta == 0 for r in reports)
