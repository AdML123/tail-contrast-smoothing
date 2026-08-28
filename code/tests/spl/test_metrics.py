import numpy as np

from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust
from hbpc.metrics import affiliation_f1


def test_events_from_binary_returns_inclusive_intervals():
    labels = np.array([0, 1, 1, 0, 1])

    assert events_from_binary(labels) == [(1, 2), (4, 4)]


def test_point_adjust_marks_whole_event_when_any_point_detected():
    labels = np.array([0, 1, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0])

    adjusted = point_adjust(pred, labels)

    np.testing.assert_array_equal(adjusted, np.array([0, 1, 1, 1, 0]))


def test_detection_delay_uses_first_hit_inside_each_event():
    labels = np.array([0, 1, 1, 0, 1, 1])
    pred = np.array([0, 0, 1, 0, 0, 0])

    assert detection_delay(pred, labels) == 1.0


def test_f1_handles_no_positive_predictions():
    assert f1(np.array([0, 0]), np.array([0, 1])) == 0.0


def test_affiliation_f1_returns_zero_when_no_predicted_events():
    labels = np.array([0, 1, 1, 0])
    pred = np.zeros_like(labels)

    assert affiliation_f1(pred, labels) == 0.0
