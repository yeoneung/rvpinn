"""Pipeline orchestrator.

  python scripts/run_all.py --config configs/paper_smoke.yaml --stage smoke
  python scripts/run_all.py --config configs/paper_full.yaml  --stage all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def step(name, cmd, log_list, allow_fail=False):
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + cmd, cwd=ROOT)
    dt = time.time() - t0
    ok = r.returncode == 0
    log_list.append({"step": name, "ok": ok, "seconds": round(dt, 1)})
    if not ok and not allow_fail:
        print(f"STEP FAILED: {name} (rc={r.returncode})")
        sys.exit(r.returncode)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["smoke", "data", "fd", "train", "evaluate",
                             "robustness", "ablations", "artifacts", "all"])
    args = ap.parse_args()
    cfg = ["--config", args.config]
    log = []

    def artifacts():
        step("consolidate", ["scripts/consolidate_results.py"] + cfg, log)
        step("tables", ["scripts/make_tables.py"], log)
        step("figures", ["scripts/make_figures.py"], log)
        step("patch manuscript", ["scripts/patch_manuscript.py"], log)
        step("compile manuscript", ["scripts/compile_manuscript.py"], log,
             allow_fail=True)
        step("reproducibility report", ["scripts/run_unit_checks.py"], log,
             allow_fail=True)
        step("experiment report", ["scripts/make_report.py"], log,
             allow_fail=True)

    if args.stage == "smoke":
        step("unit tests", ["-m", "pytest", "tests", "-x", "-q"], log)
        step("download", ["scripts/download_data.py"], log)
        step("preprocess", ["scripts/preprocess_data.py"], log)
        step("calibrate", ["scripts/calibrate_models.py"], log)
        step("expA (FD + neural)", ["scripts/run_fd_verification.py"] + cfg,
             log)
        step("train pinn_pi", ["scripts/train_pinn_pi.py"] + cfg, log)
        step("train direct_hjb", ["scripts/train_direct_hjb.py"] + cfg, log)
        step("train drl", ["scripts/train_drl.py"] + cfg, log)
        step("evaluate", ["scripts/run_real_data_evaluation.py"] + cfg, log)
        step("robustness (subset)", ["scripts/run_robustness.py"] + cfg
             + ["--families", "eff,gammaR", "--max-days", "2"], log)
        step("ablations (subset)", ["scripts/run_ablations.py"] + cfg
             + ["--parts", "a3,a5a6", "--n-seeds", "1"], log,
             allow_fail=True)
        artifacts()
    elif args.stage == "data":
        step("download", ["scripts/download_data.py"], log)
        step("preprocess", ["scripts/preprocess_data.py"], log)
        step("calibrate", ["scripts/calibrate_models.py"], log)
    elif args.stage == "fd":
        step("expA", ["scripts/run_fd_verification.py"] + cfg, log)
    elif args.stage == "train":
        step("train pinn_pi", ["scripts/train_pinn_pi.py"] + cfg, log)
        step("train direct_hjb", ["scripts/train_direct_hjb.py"] + cfg, log)
        step("train drl", ["scripts/train_drl.py"] + cfg, log)
    elif args.stage == "evaluate":
        step("evaluate", ["scripts/run_real_data_evaluation.py"] + cfg, log)
    elif args.stage == "robustness":
        step("robustness", ["scripts/run_robustness.py"] + cfg, log)
    elif args.stage == "ablations":
        step("ablations", ["scripts/run_ablations.py"] + cfg, log)
    elif args.stage == "artifacts":
        artifacts()
    elif args.stage == "all":
        step("unit tests", ["-m", "pytest", "tests", "-x", "-q"], log)
        step("download", ["scripts/download_data.py"], log)
        step("preprocess", ["scripts/preprocess_data.py"], log)
        step("calibrate", ["scripts/calibrate_models.py"], log)
        step("expA", ["scripts/run_fd_verification.py"] + cfg, log)
        step("train pinn_pi", ["scripts/train_pinn_pi.py"] + cfg, log)
        step("train direct_hjb", ["scripts/train_direct_hjb.py"] + cfg, log)
        step("train drl", ["scripts/train_drl.py"] + cfg, log)
        step("evaluate", ["scripts/run_real_data_evaluation.py"] + cfg, log)
        step("robustness", ["scripts/run_robustness.py"] + cfg, log)
        step("ablations", ["scripts/run_ablations.py"] + cfg, log)
        artifacts()

    print("\n===== pipeline summary =====")
    for l in log:
        print(f"  {'OK ' if l['ok'] else 'FAIL'} {l['step']:32s} "
              f"{l['seconds']:8.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
