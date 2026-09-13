"""Create the frozen representative-day mechanism figure and decomposition."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src.evaluation import load_region_days, run_day  # noqa: E402
from src.exp_common import regime_bundle  # noqa: E402
from src.dynamics import b_S  # noqa: E402
from src.deployment_numerics import DEPLOYMENT_VERSION  # noqa: E402
from result_validation import read_aligned_parquet  # noqa: E402


RESULTS = HERE / "results"
FIGURE = ROOT / "manuscript" / "figures" / "fig_round2_mechanism.pdf"
TEX = ROOT / "manuscript" / "generated" / "round2_mechanism.tex"
CONTROLLERS = ("learned", "reference", "m2", "m16")


def _representative_date() -> str:
    source = RESULTS / "common_daily_confirmatory_thr300_fee40_w10_n2.parquet"
    frame = read_aligned_parquet(source)
    rows = frame[(frame["method"] == "no_storage")
                 & (frame["regime"] == "winter_weekday")].copy()
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 no-storage days, found {len(rows)}")
    median = float(rows["exceed_hours"].median())
    rows["distance"] = (rows["exceed_hours"] - median).abs()
    return str(rows.sort_values(["distance", "date"]).iloc[0]["date"])


def _learned_controller(p, prof):
    from src.evaluation import NeuralController
    from src.policy_iteration import load_checkpoint

    batch_path = RESULTS / "restart_batches_v3_winter_fee40.json"
    audit = json.loads(batch_path.read_text(encoding="utf-8"))
    batch = next(row for row in audit["batch_selections"] if int(row["batch"]) == 0)
    seed = batch["selected_seed"]
    if seed is None:
        controller = lambda t, s, y, pz, ctx: 0.0
        controller.mechanism_policy_provenance = {'kind': 'zero_action', 'batch': 0}
        return controller
    selection = (ROOT / "checkpoints" / "v3_winter_fee40" / "confirmatory"
                 / "winter_weekday" / f"seed{seed}" / "validation_selection.json")
    with selection.open(encoding="utf-8") as handle:
        iteration = int(json.load(handle)["selected_iteration"])
    checkpoint = selection.parent / f"iter{iteration:02d}.pt"
    # Match the device used for the reported learned-policy evaluation.
    model = load_checkpoint(str(checkpoint), p, "cuda")
    controller = NeuralController(model, p, prof)
    controller.mechanism_policy_provenance = {
        'kind': 'trained_policy', 'batch': 0, 'seed': int(seed),
        'iteration': iteration, 'device': 'cuda',
        'checkpoint': checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
        'batch_selection_sha256': hashlib.sha256(batch_path.read_bytes()).hexdigest()}
    return controller


def _build_controller(key, p, prof):
    if key == "learned":
        return _learned_controller(p, prof)
    if key == "reference":
        from src.reference_policy import ReferenceValueController
        return ReferenceValueController(p, prof, n_grid=1025)
    # The budgeted MIQP is replayed from solver records in _run_controller.
    raise ValueError(key)


def _tex_table(summary: pd.DataFrame, date: str) -> str:
    labels = {
        "learned": "Offline learned policy",
        "reference": "No-learning reference controller",
        "m2": "Primary stochastic MIQP ($M=2$)",
        "m16": "Stochastic MIQP ($M=16$)",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (r"\caption{Cost decomposition on the protocol-selected "
         f"representative winter day ({date}).}}"),
        r"\label{tab:round2-mechanism}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Controller & Bill & Band fee & Degradation & Terminal & Total & Exceed h \\",
        r"\midrule",
    ]
    for key in CONTROLLERS:
        row = summary[summary["controller"] == key].iloc[0]
        lines.append(
            f"{labels[key]} & {row['bill']:.1f} & {row['capacity_fee']:.1f} & "
            f"{row['degradation_cost']:.1f} & {row['terminal_penalty']:.2f} & "
            f"{row['common_cost']:.1f} & {row['exceed_hours']:.2f} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def _common_inputs():
    regime = "winter_weekday"
    params, profiles, _ = regime_bundle("confirmatory", [regime])
    p = copy.deepcopy(params[regime])
    p.g_thr, p.c_step, p.w_step = 300.0, 40.0, 10.0
    prof = profiles[regime]
    date = _representative_date()
    day = next(day for day in load_region_days("confirmatory", "test")
               if day["date"] == date)
    return p, prof, date, day


class RecordedActionController:
    """Replay the exact budgeted comparator actions instead of reoptimizing."""

    def __init__(self, records, dt=.25, horizon=24.):
        ordered = records.sort_values('time')
        self.clock = np.arange(0., horizon, dt)
        if (len(ordered) != len(self.clock)
                or not np.allclose(ordered['time'], self.clock, atol=1e-12, rtol=0)):
            raise RuntimeError('recorded-action replay needs one complete decision clock')
        self.actions = np.where(ordered['has_solution'],
                                ordered['first_action_held'], 0.)
        if not np.isfinite(self.actions).all():
            raise RuntimeError('recorded-action replay has nonfinite actions')
        self.dt = dt

    def __call__(self, t, s, y, pz, ctx):
        index = int(round(t / self.dt))
        if not 0 <= index < len(self.clock) or abs(t - self.clock[index]) > 1e-10:
            raise RuntimeError('recorded-action replay called outside its decision clock')
        return float(self.actions[index])


def _reported_day(key, date):
    if key == 'learned':
        path = RESULTS / 'learned_daily_v3_winter_fee40.parquet'
        frame = read_aligned_parquet(path)
        frame = frame[frame['selected_by_batch'] & (frame['batch'] == 0)]
    elif key == 'reference':
        path = RESULTS / 'round2_reference_daily.parquet'
        frame = read_aligned_parquet(path)
        frame = frame[frame['fee_per_hour'] == 40]
    elif key in ('m2', 'm16'):
        name = ('common_daily_confirmatory_thr300_fee40_w10_n2.parquet' if key == 'm2'
                else 'common_daily_confirmatory_thr300_fee40_w10_n16_r2_stream7301_pool16_tl5.parquet')
        path = RESULTS / name
        frame = read_aligned_parquet(path)
        frame = frame[frame['method'] == 'stochastic_two_stage_exact_band_miqp']
    else:
        raise ValueError(key)
    selected = frame[frame['date'].astype(str) == str(date)]
    if len(selected) != 1:
        raise RuntimeError(f'mechanism needs exactly one reported row for {key}/{date}')
    return selected.iloc[0], path


def _check_reported_economics(result, reported):
    for name in ('bill', 'capacity_fee', 'degradation_cost', 'terminal_penalty',
                 'common_cost', 'exceed_hours', 'throughput_kwh', 'terminal_soc'):
        if not np.isclose(result[name], reported[name], atol=1e-7, rtol=0):
            raise RuntimeError(f'representative-day plot disagrees with reported {name}')
    if result['exact_action_change_events'] != 0:
        raise RuntimeError('representative-day replay changed a held action at the safety gate')


def _run_controller(key: str) -> None:
    p, prof, date, day = _common_inputs()
    reported, daily_path = _reported_day(key, date)
    source_files = [daily_path]
    if key in ('m2', 'm16'):
        solves_path = RESULTS / daily_path.name.replace('common_daily_', 'common_solves_')
        solves = read_aligned_parquet(solves_path)
        records = solves[(solves['date'].astype(str) == str(date))
                         & (solves['method'] == 'stochastic_two_stage_exact_band_miqp')]
        controller = RecordedActionController(records, p.dt_ctrl, p.T)
        source_files.append(solves_path)
    else:
        controller = _build_controller(key, p, prof)
    print(f"running {key} on {date}", flush=True)
    result = run_day(controller, day, p, prof, collect_traj=True)
    _check_reported_economics(result, reported)
    summary = {
        "controller": key,
        "date": str(date),
        "deployment_version": DEPLOYMENT_VERSION,
        "reported_economics_verified": True,
        "trajectory_source": ("replay of recorded MIQP actions" if key in ('m2', 'm16')
                              else "same-policy reevaluation checked against reported day"),
        "policy_provenance": getattr(controller, 'mechanism_policy_provenance', {}),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        **{name: result[name] for name in (
            "bill", "capacity_fee", "degradation_cost",
            "terminal_penalty", "common_cost", "exceed_hours",
            "throughput_kwh", "terminal_soc")},
        # Replay latency is not a new measurement of online solver cost.
        "latency_mean_ms": float(reported['latency_mean_ms']),
    }
    trajectory_path = RESULTS / f"round2_mechanism_{key}_trajectory.parquet"
    pd.DataFrame(result["traj"]).to_parquet(trajectory_path, index=False)
    summary['trajectory_sha256'] = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
    (RESULTS / f"round2_mechanism_{key}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def _soc_plot_coordinates(trajectory, terminal_soc, p):
    """Recover control-boundary SoC from the retained first-substep records.

    run_day stores G and a at t, but its s entry is after the first five-minute
    integration substep. Correct only the figure's time coordinates; preserve
    the original trajectory and all economic outputs.
    """
    t = np.asarray(trajectory['t'], dtype=float)
    drift = b_S(np.asarray(trajectory['a'], dtype=float), p)
    start_soc = np.asarray(trajectory['s'], dtype=float) - p.dt_sim * drift
    expected_clock = np.arange(0.0, p.T, p.dt_ctrl)
    if t.shape != expected_clock.shape or not np.allclose(t, expected_clock, atol=1e-12, rtol=0):
        raise RuntimeError('mechanism trajectory has an unexpected control clock')
    endpoint_soc = np.r_[start_soc, terminal_soc]
    if (not np.isclose(start_soc[0], p.s0, atol=1e-10, rtol=0)
            or not np.allclose(endpoint_soc[1:], start_soc + p.dt_ctrl * drift,
                               atol=1e-10, rtol=0)):
        raise RuntimeError('mechanism SoC records do not match the held-action dynamics')
    return np.r_[t, p.T], endpoint_soc


def _assemble() -> None:
    p, _, date, _ = _common_inputs()
    outputs = {}
    summary_rows = []
    trajectory_rows = []
    for key in CONTROLLERS:
        summary_path = RESULTS / f"round2_mechanism_{key}_summary.json"
        trajectory_path = RESULTS / f"round2_mechanism_{key}_trajectory.parquet"
        if not summary_path.exists() or not trajectory_path.exists():
            raise FileNotFoundError(f"run controller {key} before assembly")
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        if (row.get('deployment_version') != DEPLOYMENT_VERSION
                or row.get('reported_economics_verified') is not True
                or row.get('date') != str(date)):
            raise RuntimeError('representative-day artifacts need corrected, source-checked regeneration')
        for name, digest in row['source_sha256'].items():
            if hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('reported mechanism source changed after trajectory generation')
        if hashlib.sha256(trajectory_path.read_bytes()).hexdigest() != row['trajectory_sha256']:
            raise RuntimeError('mechanism trajectory changed after its economic check')
        summary_rows.append(row)
        trajectory = pd.read_parquet(trajectory_path)
        outputs[key] = {name: trajectory[name].to_numpy()
                        for name in trajectory.columns}
        trajectory.insert(0, "controller", key)
        trajectory_rows.append(trajectory)

    summary = pd.DataFrame(summary_rows)
    trajectory = pd.concat(trajectory_rows, ignore_index=True)
    summary.to_csv(RESULTS / "round2_mechanism_day_summary.csv", index=False)
    trajectory.to_parquet(
        RESULTS / "round2_mechanism_day_trajectory.parquet", index=False)
    metadata = {
        "date": date,
        "selection_rule": (
            "closest observed no-storage exceedance duration to the median; "
            "earliest date breaks ties"),
        "fee_per_hour": 40.0,
        "threshold_kw": 300.0,
        "scenario_count": 16,
        "scenario_stream": 7301,
        "scenario_pool_size": 16,
        "time_limit_s": 5.0,
        "learned_batch": 0,
        "primary_scenario_count": 2,
        "primary_scenario_stream": 4101,
        "primary_curve_scope": "added during numerical review from already reported primary actions",
        "raw_soc_time_offset_hours": p.dt_sim,
        "figure_soc_time_basis": "control boundaries reconstructed from retained substep records",
    }
    (RESULTS / "round2_mechanism_day_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    styles = {
        "learned": ("Offline learned", "#0072B2", "-"),
        "reference": ("No-learning reference", "#D55E00", "--"),
        "m2": ("Primary stochastic MIQP ($M=2$)", "#CC79A7", ":"),
        "m16": ("Stochastic MIQP ($M=16$, 5 s)", "#009E73", "-."),
    }
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.1), sharex=True)
    for key, (label, color, linestyle) in styles.items():
        tr = outputs[key]
        step_clock = np.r_[tr['t'], p.T]
        axes[0].step(step_clock, np.r_[tr['G'], tr['G'][-1]], where='post',
                     color=color, ls=linestyle, lw=1.4, label=label)
        axes[1].step(step_clock, np.r_[tr['a'], tr['a'][-1]], where='post',
                     color=color, ls=linestyle, lw=1.4)
        terminal_soc = float(summary.loc[summary['controller'] == key, 'terminal_soc'].iloc[0])
        soc_clock, soc_values = _soc_plot_coordinates(tr, terminal_soc, p)
        axes[2].plot(soc_clock, soc_values, color=color, ls=linestyle,
                     lw=1.4)
    tr0 = outputs["learned"]
    axes[0].step(np.r_[tr0['t'], p.T], np.r_[tr0['N'], tr0['N'][-1]],
                 where='post', color="0.55", lw=0.9,
                 alpha=0.8, label="Uncontrolled net load")
    axes[0].axhline(p.g_thr, color="black", lw=1.0, ls=":",
                    label="Band threshold")
    axes[1].axhline(0.0, color="0.5", lw=0.7)
    axes[2].axhline(p.s_min, color="0.6", lw=0.7, ls=":")
    axes[2].axhline(p.s_max, color="0.6", lw=0.7, ls=":")
    axes[0].set_ylabel("Grid exchange (kW)")
    axes[1].set_ylabel("Battery power (kW)")
    axes[2].set_ylabel("SoC")
    axes[2].set_xlabel("Hour")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, fontsize=8,
               loc="upper center")
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.18)
    fig.tight_layout(rect=(0., 0., 1., .89))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)
    TEX.write_text(_tex_table(summary, date), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(FIGURE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=CONTROLLERS)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.controller:
        _run_controller(args.controller)
    if args.assemble:
        _assemble()
    if not args.controller and not args.assemble:
        parser.error("specify --controller or --assemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
