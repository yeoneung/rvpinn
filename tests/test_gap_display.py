"""Do not confuse a unitless gap of 7.8 with 7.8 percent."""
import pytest

from experiments.result_validation import format_gap_percent


@pytest.mark.parametrize('value,expected', [(0., r'0.00\%'), (.01, r'1.00\%'),
                                          (7.81233, r'781.23\%'),
                                          (float('nan'), '--'), (float('inf'), '--'),
                                          (1e20, '--')])
def test_gap_ratio_display(value, expected):
    assert format_gap_percent(value) == expected


def test_negative_gap_is_not_displayed_as_a_solver_certificate():
    with pytest.raises(ValueError, match='negative'):
        format_gap_percent(-.01)
