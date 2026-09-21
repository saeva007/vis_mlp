#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "gate_study.json").read_text(encoding="utf-8"))
RUN_ROOT = Path(CFG["project_root"]) / "runs"
NAMES = {"A": "A_linear_frozen", "B": "B_mlp_frozen", "C": "C_mlp_e2e"}


def metrics_delta(left, right):
    keys = ["AUROC", "AP", "balanced_accuracy", "precision", "recall", "Brier"]
    return {key: right[key] - left[key] for key in keys}


results = {key: json.loads((RUN_ROOT / name / "test_results.json").read_text(encoding="utf-8")) for key, name in NAMES.items()}
margins = list(results["A"]["test"]["margin_metrics"])
summary = {
    "experiments": results,
    "deltas": {
        "B_minus_A_head_capacity": {
            margin: metrics_delta(
                results["A"]["test"]["margin_metrics"][margin],
                results["B"]["test"]["margin_metrics"][margin],
            ) for margin in margins
        },
        "C_minus_B_encoder_finetuning": {
            margin: metrics_delta(
                results["B"]["test"]["margin_metrics"][margin],
                results["C"]["test"]["margin_metrics"][margin],
            ) for margin in margins
        },
        "margin_gain_vs_all": {
            key: {
                margin: metrics_delta(
                    result["test"]["margin_metrics"]["all"],
                    result["test"]["margin_metrics"][margin],
                ) for margin in margins if margin != "all"
            } for key, result in results.items()
        },
    },
}
out = Path(CFG["project_root"]) / "gate_separability_summary.json"
tmp = out.with_suffix(".json.tmp")
tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
tmp.replace(out)
print(out)
