"""Measure selected-checkpoint loading as a separate deployment overhead."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
sys.path.insert(0, str(ROOT))

from src.exp_common import regime_bundle  # noqa: E402
from src.policy_iteration import load_checkpoint  # noqa: E402


TAGS = {
    20: "v3_winter_fee20",
    40: "v3_winter_fee40",
    80: "v3_winter_fee80",
}


def main() -> int:
    rows = []
    params, _, _ = regime_bundle("confirmatory", ["winter_weekday"])
    for fee, tag in TAGS.items():
        p = params["winter_weekday"]
        p.g_thr, p.c_step, p.w_step = 300.0, float(fee), 10.0
        audit = json.loads((RESULTS / f"restart_batches_{tag}.json")
                           .read_text(encoding="utf-8"))
        restart_by_seed = {int(row["seed"]): row
                           for row in audit["restarts"]}
        for selection in audit["batch_selections"]:
            batch = int(selection["batch"])
            seed = selection.get("selected_seed")
            if seed is None:
                rows.append({
                    "fee": fee, "batch": batch,
                    "selected_kind": "zero_action_fallback",
                    "selected_seed": None, "model_loading_s": 0.0,
                })
                continue
            checkpoint = restart_by_seed[int(seed)]["checkpoint"]
            started = time.perf_counter()
            model = load_checkpoint(checkpoint, p, "cuda")
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rows.append({
                "fee": fee, "batch": batch,
                "selected_kind": "trained_policy",
                "selected_seed": int(seed),
                "model_loading_s": float(elapsed),
            })
            del model
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "confirmatory_model_loading.csv", index=False)
    summary = {
        "definition": (
            "wall time from checkpoint read through CUDA synchronization; "
            "zero for a retained zero-action fallback"),
        "n_selected_trained_models": int(
            (frame["selected_kind"] == "trained_policy").sum()),
        "mean_model_loading_s": float(frame["model_loading_s"].mean()),
        "max_model_loading_s": float(frame["model_loading_s"].max()),
    }
    (RESULTS / "confirmatory_model_loading.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
