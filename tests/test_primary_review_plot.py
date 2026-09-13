"""Primary-result plots can be regenerated before the factorial jobs finish."""
import pandas as pd
import pytest
from matplotlib.axes import Axes

from experiments import make_v3_figures as plotting


def test_cost_latency_plot_does_not_require_factorial_results(tmp_path, monkeypatch):
    methods = (
        'no_storage', 'validation_tuned_price_rule', 'convex_envelope_mpc',
        'deterministic_exact_band_miqp', 'stochastic_two_stage_exact_band_miqp',
        'offline_value_policy',
    )
    rows = [dict(fee=fee, method=method, mean_common_cost=1000. + 10. * index,
                 mean_latency_mean_ms=float(index * 100 + .00025))
            for fee in (20, 40, 80) for index, method in enumerate(methods)]
    pd.DataFrame(rows).to_csv(tmp_path / 'confirmatory_method_summary.csv', index=False)
    monkeypatch.setattr(plotting, 'RESULTS', tmp_path)
    monkeypatch.setattr(plotting, 'FIGURES', tmp_path / 'figures')
    actual_x, actual_axes, scatter = [], [], Axes.scatter
    def record_x(self, x, y, *args, **kwargs):
        actual_x.append(x)
        actual_axes.append(self)
        return scatter(self, x, y, *args, **kwargs)
    monkeypatch.setattr(Axes, 'scatter', record_x)
    plotting.make_cost_latency()
    assert actual_x[0] == .00025  # no artificial 1e-3-ms plotting floor
    assert actual_axes[0].get_shared_x_axes().joined(actual_axes[0], actual_axes[6])
    assert actual_axes[0].get_shared_x_axes().joined(actual_axes[0], actual_axes[12])
    assert (tmp_path / 'figures' / 'fig_cost_latency.pdf').read_bytes().startswith(b'%PDF')
    assert (tmp_path / 'figures' / 'fig_cost_latency.png').is_file()
    assert not (tmp_path / 'figures' / 'fig_factorial.pdf').exists()
    assert plotting.matplotlib.rcParams['pdf.fonttype'] == 42


def test_log_latency_plot_rejects_nonpositive_data_instead_of_moving_points(tmp_path, monkeypatch):
    pd.DataFrame([dict(mean_latency_mean_ms=0.)]).to_csv(
        tmp_path / 'confirmatory_method_summary.csv', index=False)
    monkeypatch.setattr(plotting, 'RESULTS', tmp_path)
    monkeypatch.setattr(plotting, 'FIGURES', tmp_path / 'figures')
    with pytest.raises(ValueError, match='finite positive latencies'):
        plotting.make_cost_latency()
