#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/public/home/putianshu/vis_mlp/gate_separability_20260915"
jobs = {}
names = {"A": "gate_A_linear", "B": "gate_B_mlp", "C": "gate_C_e2e"}
for experiment, name in names.items():
    output = subprocess.check_output([
        "sbatch", "--parsable", "--job-name", name,
        f"{REMOTE_ROOT}/scripts/sub_gate.slurm", experiment,
    ], text=True).strip()
    jobs[experiment] = output.split(";")[0]
dependency = ":".join(jobs.values())
aggregate = subprocess.check_output([
    "sbatch", "--parsable", "--job-name", "gate_aggregate",
    f"--dependency=afterok:{dependency}",
    f"{REMOTE_ROOT}/scripts/sub_aggregate.slurm",
], text=True).strip().split(";")[0]
record = {"training_jobs": jobs, "aggregate_job": aggregate}
(Path(REMOTE_ROOT) / "submitted_jobs.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record))
