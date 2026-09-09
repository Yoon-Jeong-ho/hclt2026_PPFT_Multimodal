from scripts.generate_predictions import score


def test_prediction_scoring_routes_squad_mcqa_and_open_vqa() -> None:
    squad = {
        "source": "squad",
        "final_answer": "Paris",
        "answer_aliases": ["Paris"],
        "options": None,
    }
    assert score(squad, "The answer is Paris.")["relaxed_result"]

    mcqa = {
        "source": "Pri-DDX",
        "final_answer": "kidney",
        "answer_aliases": ["B"],
        "options": ["heart", "kidney", "lung"],
        "gold_option_index": 1,
    }
    assert score(mcqa, "B. kidney is correct; heart is not.")["relaxed_result"]
    assert not score(mcqa, "Heart is tempting, but the answer is kidney.")["relaxed_result"]

    vqa = {
        "source": "VQA-RAD",
        "final_answer": "CT",
        "answer_aliases": ["computed tomography"],
        "options": None,
    }
    assert score(vqa, "The answer is computed tomography.")["relaxed_result"]
