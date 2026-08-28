import numpy as np

from hbpc.mars import (
    ewma_score,
    jaccard_indices,
    mars_abs_score,
    mars_rel_score,
    shift_alarm_indices,
    tail_score,
    top_n_indices,
)


def test_top_n_indices_ignores_nan_and_returns_descending_score_order():
    scores = np.array([0.0, np.nan, 5.0, 3.0, 5.0])
    assert top_n_indices(scores, 3).tolist() == [2, 4, 3]


def test_shift_alarm_indices_drops_indices_that_shift_past_length():
    shifted = shift_alarm_indices(np.array([1, 4]), shift=2, length=5)
    assert shifted.tolist() == [3]


def test_jaccard_indices_handles_empty_sets():
    assert jaccard_indices([], []) == 1.0
    assert jaccard_indices([1, 2], []) == 0.0
    assert jaccard_indices([1, 2], [2, 3]) == 1 / 3


def test_tail_score_uses_future_window_and_nan_edges():
    r = np.array([1, 1, 9, 7, 6, 5], dtype=float)
    scores = tail_score(r, k=3)
    assert scores[2] == 6.0
    assert np.isnan(scores[-3:]).all()


def test_mars_abs_rewards_worsening_and_suppresses_recovery():
    recovering = np.array([1, 1, 9, 1, 1, 1], dtype=float)
    worsening = np.array([1, 1, 9, 7, 6, 5], dtype=float)
    rec = mars_abs_score(recovering, k=3, alpha=1.0)
    wor = mars_abs_score(worsening, k=3, alpha=1.0)
    assert rec[2] == 0.0
    assert wor[2] > 8.0


def test_mars_rel_is_tail_scaled_by_relative_momentum():
    r = np.array([1, 1, 4, 4, 4, 4], dtype=float)
    scores = mars_rel_score(r, k=3, alpha=1.0)
    assert scores[2] > tail_score(r, k=3)[2]


def test_ewma_score_matches_manual_recursion():
    r = np.array([0.0, 10.0, 0.0], dtype=float)
    scores = ewma_score(r, alpha=0.5)
    assert np.allclose(scores, [0.0, 5.0, 2.5])
