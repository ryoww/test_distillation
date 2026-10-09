from src.lm_config import LMConfig, create_lm


def test_clients_never_reuse_cached_answers_across_models():
    lm = create_lm(LMConfig(model="student", api_base="http://127.0.0.1:7600/v1"))
    assert lm.cache is False
