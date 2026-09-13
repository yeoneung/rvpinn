"""Export compact, source-backed parameter tables for the supplement."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "generated" / "round2_parameter_tables.tex"


def _load_yaml(name: str):
    with (ROOT / "configs" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _selected_scale(tag: str, regime: str, seed: int) -> float:
    manifest = json.loads((ROOT / "data" / "checkpoint_metadata.json")
                          .read_text(encoding="utf-8"))["checkpoints"]
    prefix = f"{tag}/confirmatory/{regime}/seed{seed}/"
    scales = {float(record["out_scale"]) for key, record in manifest.items()
              if key.startswith(prefix)}
    if len(scales) != 1:
        raise RuntimeError(f"missing or inconsistent checkpoint scales: {prefix}")
    return scales.pop()


def main() -> int:
    model = _load_yaml("model.yaml")
    confirmatory = _load_yaml("confirmatory.yaml")
    with (ROOT / "data" / "calibrated_parameters.json").open(
            encoding="utf-8") as handle:
        calibrated = json.load(handle)["confirmatory"]

    battery = model["battery"]
    horizon = model["horizon"]
    microgrid = model["microgrid"]
    cost = model["cost"]
    coeff = calibrated["coefficients"]
    network = confirmatory["network"]
    evaluation = confirmatory["policy_evaluation"]
    selection = confirmatory["policy_selection"]

    fixed_rows = [
        ("Battery energy; charge/discharge limits",
         f"{battery['E_max_kwh']:.0f} kWh; "
         f"{battery['a_c_max_kw']:.0f}/{battery['a_d_max_kw']:.0f} kW"),
        ("SoC bounds; initial/target SoC",
         f"[{battery['s_min']:.2f}, {battery['s_max']:.2f}]; "
         f"{battery['s0']:.2f}/{battery['s_target']:.2f}"),
        ("Charge/discharge efficiency; taper width",
         f"{battery['eta_c']:.2f}/{battery['eta_d']:.2f}; "
         f"{battery['delta_s']:.2f} SoC"),
        ("Horizon; control/simulation interval",
         f"{horizon['T_hours']:.0f} h; "
         f"{horizon['control_dt_min']:.0f}/{horizon['sim_dt_min']:.0f} min"),
        ("Base power; renewable scaling",
         f"{microgrid['P_base_kw']:.0f} kW; "
         f"$\\gamma_R={microgrid['gamma_R']:.1f}$"),
        ("Export fraction; throughput cost",
         f"$\\alpha_s={cost['alpha_s']:.2f}$; "
         f"$\\lambda_1={coeff['lam1']:.4f}$ EUR/kWh"),
        ("Terminal coefficient; terminal target",
         f"$\\lambda_T={coeff['lam_T']:,.2f}$ EUR/SoC$^2$; "
         f"$s_{{\\rm tar}}={battery['s_target']:.2f}$"),
        ("Tariff threshold; fee anchors; smooth widths",
         "300 kW; 20/40/80 EUR/h; 5/10/20 kW"),
        ("Hard deployment search; quadrature",
         "1,025-point structural grid; $3\\times3$ Gauss--Hermite"),
        ("Primary online-MIQP budget; target gap",
         "5 s per solve; 1\\%"),
        ("Solver feasibility tolerance; binary separation",
         "$10^{-9}$; $10^{-5}$ kW"),
        ("Scenario sensitivity",
         "$M=2,8,16$; three nested streams; pool size 16"),
        ("Network; Fourier harmonics",
         f"{network['hidden_layers']} layers $\\times$ "
         f"{network['width']}; 1, 2, 3"),
        ("Training per PI round",
         f"at most {evaluation['adam_max_steps']:,} Adam and "
         f"{evaluation['lbfgs_max_iter']} L-BFGS steps"),
        ("PI rounds; selection margin",
         f"{confirmatory['policy_iteration']['max_iterations']}; "
         f"{selection['min_mean_improvement_eur_per_day']:.2f} EUR/day"),
    ]

    scale_lookup = {}
    tag_lookup = {
        ("winter", 20): ("v3_winter_fee20", "winter_weekday"),
        ("winter", 40): ("v3_winter_fee40", "winter_weekday"),
        ("winter", 80): ("v3_winter_fee80", "winter_weekday"),
        ("spring", 40): ("v3_spring_fee40", "spring_weekday"),
        ("summer", 40): ("v3_summer_fee40", "summer_weekday"),
        ("autumn", 40): ("v3_autumn_fee40", "autumn_weekday"),
    }
    for key, (tag, regime) in tag_lookup.items():
        # beta is fixed before training, not selected on validation/test.
        # Verify that all archived restarts and iterations share that scale.
        manifest = json.loads((ROOT / "data" / "checkpoint_metadata.json")
                              .read_text(encoding="utf-8"))["checkpoints"]
        prefix = f"{tag}/confirmatory/{regime}/"
        scales = {float(record["out_scale"]) for path, record in manifest.items()
                  if path.startswith(prefix)}
        if len(scales) != 1:
            raise RuntimeError(f"nonconstant output scale in {prefix}")
        scale_lookup[key] = scales.pop()

    ou_rows = []
    for season in ("winter", "spring", "summer", "autumn"):
        regime = calibrated["regimes"][f"{season}_weekday"]
        ou_rows.append((
            season.capitalize(), regime["kappa_y"], regime["sigma_y"],
            regime["y_min"], regime["y_max"],
            regime["kappa_p"], regime["sigma_p"],
            regime["p_min"], regime["p_max"], regime["rho"],
            regime["n_train_hours"], scale_lookup[(season, 40)]))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Fixed physical, economic, controller, and computational parameters. All values were fixed from training data or the frozen experimental protocol.}",
        r"\label{tab:supp-fixed-parameters}",
        r"\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
    ]
    lines.extend(f"{name} & {value} \\\\" for name, value in fixed_rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Weekday OU calibration and central-fee neural output scale $\beta$ by season. Mean-reversion rates are per hour; diffusion scales are in kW/$\sqrt{\mathrm{h}}$ and EUR/kWh/$\sqrt{\mathrm{h}}$; $\beta$ is in EUR/h.}",
        r"\label{tab:supp-ou-parameters}",
        r"\begin{tabularx}{\linewidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xrr}",
        r"\toprule",
        r"Season & Load error $(\kappa_y,\sigma_y,[y_{\min},y_{\max}])$ & Price error $(\kappa_p,\sigma_p,[p_{\min},p_{\max}])$ & $\rho$ & $\beta$ (EUR/h) \\",
        r"\midrule",
    ])
    for row in ou_rows:
        season, ky, sy, ymin, ymax, kp, sp, pmin, pmax, rho, hours, scale = row
        lines.append(
            f"{season} & ({_fmt(ky)}, {_fmt(sy, 2)}, "
            f"[{_fmt(ymin, 1)}, {_fmt(ymax, 1)}]) & "
            f"({_fmt(kp)}, {_fmt(sp, 5)}, "
            f"[{_fmt(pmin, 4)}, {_fmt(pmax, 4)}]) & "
            f"{_fmt(rho, 3)} & {_fmt(scale, 2)} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        ("The winter, spring, summer, and autumn weekday calibrations use "
         + ", ".join(f"{int(row[-2]):,}" for row in ou_rows)
         + " training hours, respectively. "),
        r"For the two outer winter fees, the neural output scales are "
        f"{scale_lookup[('winter', 20)]:.2f} (20 EUR/h) and "
        f"{scale_lookup[('winter', 80)]:.2f} (80 EUR/h). "
        r"The hard-band binary separation is $10^{-5}$ kW. State-domain "
        r"reflection bounds, hourly profiles, and every remaining calibrated "
        r"coefficient are stored in the machine-readable parameter file.",
        "",
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
