import pytest

from experiments.run_common_comparators import select_evaluation_days


def test_day_budget_applies_to_each_requested_regime():
    days = [
        {'date': '2019-01-02', 'regime': 'winter_weekday'},
        {'date': '2019-01-03', 'regime': 'winter_weekday'},
        {'date': '2019-01-04', 'regime': 'winter_weekday'},
        {'date': '2019-04-01', 'regime': 'spring_weekday'},
        {'date': '2019-04-02', 'regime': 'spring_weekday'},
        {'date': '2019-04-03', 'regime': 'spring_weekday'},
    ]
    selected = select_evaluation_days(days[::-1],
                                      {'winter_weekday', 'spring_weekday'}, 2)
    assert [day['date'] for day in selected] == [
        '2019-01-02', '2019-01-03', '2019-04-01', '2019-04-02']
    assert select_evaluation_days(days, {'winter_weekday'}, 2) == days[:2]
    assert select_evaluation_days(days, {'winter_weekday'}, None) == days[:3]
    with pytest.raises(ValueError):
        select_evaluation_days(days, {'winter_weekday'}, 0)
