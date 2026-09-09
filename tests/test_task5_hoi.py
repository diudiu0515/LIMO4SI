import pytest

from limo4si.task5_hoi import analyze_window, body_relative_vector, relation_label, synchronized_indices


def test_body_relative_identity_frame():
    relative = body_relative_vector([1, 2, 3], [0, 0, 0], [2, 2, 5])
    assert relative == pytest.approx([1, 0, 2])
    assert relation_label(relative) == "right-front"


def test_synchronization_is_exact_and_ordered_by_human_stream():
    assert synchronized_indices(["t1", "t2", "t3"], ["t3", "t1"]) == [(0, 1, "t1"), (2, 0, "t3")]


def test_window_reports_annotation_scope_and_distance_change():
    result = analyze_window(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 2], [0, 0, 3]], fps=1,
    )
    assert result["distance_change_m"] == pytest.approx(2)
    assert result["object_displacement_m"] == pytest.approx(2)
    assert "no gaze/contact claim" in result["evidence_scope"]
