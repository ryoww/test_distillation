from scripts.replay_failed import convert_id


def test_instance_ids_match_the_evaluator_format():
    assert convert_id(4035) == "prob_4035"
    assert convert_id(7) == "prob_007"
