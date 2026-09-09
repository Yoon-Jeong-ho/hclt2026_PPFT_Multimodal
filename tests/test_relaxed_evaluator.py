from ppft_multimodal.evaluation.normalization import normalize_medical_answer
from ppft_multimodal.evaluation.relaxed_qa import relaxed_containment, relaxed_first_match, score_mcqa
from ppft_multimodal.evaluation.squad import score_squad

OPTIONS = ["heart", "lung", "kidney failure", "liver"]


def test_gold_first_is_correct_and_distractor_first_is_wrong() -> None:
    assert score_mcqa("The answer is kidney failure. Not lung.", OPTIONS, 2)["relaxed_correct"]
    assert not score_mcqa("Lung is tempting, but kidney failure is correct.", OPTIONS, 2)["relaxed_correct"]


def test_mcqa_primary_requires_option_text_but_keeps_legacy_label_diagnostics() -> None:
    result = score_mcqa("B", OPTIONS, 1)
    assert result["relaxed_correct"] is False
    assert result["strict_correct"] is False
    assert result["option_text_correct"] is False
    assert result["legacy_relaxed_correct"] is True
    assert result["label_only_correct"] is True


def test_mcqa_explicit_legacy_primary_mode_remains_available() -> None:
    result = score_mcqa("B", OPTIONS, 1, allow_labels=True)
    assert result["relaxed_correct"] is True
    assert result["strict_correct"] is False


def test_mcqa_ignores_preceding_labels_when_scoring_option_text() -> None:
    correct_label = score_mcqa("B. lung", OPTIONS, 1)
    wrong_label = score_mcqa("B. kidney failure", OPTIONS, 2)
    assert correct_label["relaxed_correct"] is True
    assert correct_label["option_text_correct"] is True
    assert correct_label["label_only_correct"] is False
    assert wrong_label["relaxed_correct"] is True
    assert wrong_label["option_text_correct"] is True
    assert wrong_label["legacy_relaxed_correct"] is False


def test_mcqa_earliest_conflicting_real_option_text_still_wins() -> None:
    result = score_mcqa("Lung is tempting, but kidney failure is correct.", OPTIONS, 2)
    assert result["relaxed_correct"] is False
    assert result["predicted_index"] == 1


def test_mcqa_literal_single_letter_option_text_is_valid() -> None:
    result = score_mcqa("B", ["A", "B", "C"], 1)
    assert result["relaxed_correct"] is True
    assert result["strict_correct"] is True
    assert result["option_text_correct"] is True


def test_mcqa_letter_alias_cannot_reintroduce_label_credit() -> None:
    result = score_mcqa(
        "B",
        ["heart", "lung", "kidney"],
        1,
        option_aliases=[[], ["B"], []],
    )
    assert result["relaxed_correct"] is False
    assert result["option_text_correct"] is False
    assert result["legacy_relaxed_correct"] is True
    assert result["label_only_correct"] is True


def test_mcqa_strict_requires_exact_first_line_option_text() -> None:
    exact = score_mcqa("kidney failure\nAdditional explanation.", OPTIONS, 2)
    prose = score_mcqa("The answer is kidney failure.", OPTIONS, 2)
    assert exact["strict_correct"] is True
    assert prose["strict_correct"] is False
    assert prose["relaxed_correct"] is True


def test_label_inside_word_does_not_match() -> None:
    result = relaxed_first_match("cat", ["zero", "one", "two", "three"])
    assert not result.valid


def test_english_article_is_not_mistaken_for_option_a() -> None:
    result = relaxed_first_match("This is a kidney failure case.", OPTIONS)
    assert result.predicted_index == 2


def test_korean_label_copula_matches_but_topic_particle_does_not() -> None:
    result = relaxed_first_match("정답은 B다. A는 오답이다.", ["one", "two"])
    assert result.predicted_index == 1


def test_longest_overlapping_option_at_same_position_wins() -> None:
    result = relaxed_first_match("heart failure", ["heart", "heart failure", "lung"])
    assert result.predicted_index == 1


def test_equal_span_tie_is_ambiguous() -> None:
    result = relaxed_first_match("same", ["same", "same"])
    assert result.ambiguous and not result.valid

    score = score_mcqa("same", ["same", "same"], 0)
    assert score["ambiguous"] is True
    assert score["relaxed_correct"] is False
    assert score["option_text_correct"] is False


def test_unicode_case_punctuation_normalization_and_containment() -> None:
    assert normalize_medical_answer(" ＹＥＳ！ ") == "yes"
    assert relaxed_containment("Answer: CT.", ["ct"])
    assert relaxed_first_match("정답은 ＫＩＤＮＥＹ ＦＡＩＬＵＲＥ.", OPTIONS).predicted_index == 2


def test_squad_standard_and_relaxed_scores() -> None:
    scores = score_squad("The answer is the Eiffel Tower in Paris", ["Eiffel Tower"])
    assert scores["exact_match"] == 0
    assert 0 < scores["token_f1"] < 1
    assert scores["relaxed_correct"]
