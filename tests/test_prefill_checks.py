import pytest
from pacer.prefill_checks import check_summary


def sample(margin, error, equal):
    return dict(
        ref_margin=margin,
        max_abs=error,
        argmax_equal=equal,
        ref_absmax=10,
        hit_blocks=1,
        n_blocks=2,
    )


def test_two_logits_can_move_in_opposite_directions():
    # [1.5, 0] -> [0.5, 1]: error 1, margin 1.5, flipped winner.
    s = check_summary([sample(1.5, 1, False), sample(3, 1, True)])
    assert s["argmax_equal_frac"] == 0.5
    assert s["clear_margin_n"] == 1
    assert s["argmax_equal_frac_clear_margin"] == 1
    assert s["protocol_version"] == 2


@pytest.mark.parametrize("margin", [0, 1, 2])
def test_boundary_is_not_certified(margin):
    s = check_summary([sample(margin, 1, True)])
    assert s["clear_margin_n"] == 0
    assert s["argmax_equal_frac_clear_margin"] is None


def test_zero_error_and_empty_sample():
    assert check_summary([]) == {"n": 0}
    assert check_summary([sample(1, 0, True)])["clear_margin_n"] == 1


def test_bitwise_fraction_counts_zero_error_only():
    s = check_summary([sample(1, 0, True), sample(1, 1e-7, True)])
    assert s["bitwise_equal_frac"] == 0.5
    assert s["argmax_equal_frac"] == 1.0
