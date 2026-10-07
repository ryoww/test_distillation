from src.modules import analyze_instance_structure


def test_matrix_field_is_summarised_by_shape_not_dumped():
    instance = {"scenario_returns": [[0.01 * j for j in range(1200)] for _ in range(300)], "n": 3}
    text = analyze_instance_structure(instance)
    assert "matrix 300x1200" in text
    assert "scenario_returns[i][j]" in text
    assert len(text) < 2000


def test_flat_list_sample_is_unchanged():
    text = analyze_instance_structure({"weights": [1, 2, 3, 4]})
    assert "Sample: [1, 2, 3]" in text
