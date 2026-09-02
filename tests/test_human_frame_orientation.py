import pytest

from limo4si.human_frame import HumanFrame, apply_forward_sign


def test_audited_forward_flip_changes_only_forward_axis():
    frame = HumanFrame(
        origin=(1.0, 2.0, 3.0),
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
    )
    flipped = apply_forward_sign(frame, -1)
    assert flipped.origin == frame.origin
    assert flipped.right == frame.right
    assert flipped.up == frame.up
    assert flipped.forward == (0.0, 0.0, -1.0)
    assert flipped.world_to_human((1.0, 2.0, 4.0))[2] == -1.0


def test_forward_sign_rejects_invalid_calibration():
    frame = HumanFrame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    with pytest.raises(ValueError):
        apply_forward_sign(frame, 0)
