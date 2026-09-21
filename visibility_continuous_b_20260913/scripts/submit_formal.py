#!/usr/bin/env python3
"""Submit exactly one exclusive-node job for each requested route/seed."""
import json
import subprocess
import sys
from pathlib import Path

root = Path("/public/home/putianshu/vis_mlp/visibility_continuous_b_20260913")
selection = json.load(open(Path(sys.argv[1])))
if selection.get("status") != "selected":
    raise SystemExit("benchmark resource selection is incomplete")
dcu = int(selection["dcu_count"])
allocated_dcu = 4
batch = int(selection["batch_size_per_rank"])
workers = int(selection["num_workers_total"])
tasks = [("R2b", s) for s in (1, 2, 3)] + [("R3b", s) for s in (1, 2, 3)] + [("R4b", s) for s in (1, 2, 3, 4)]
idle_nodes = subprocess.check_output(
    ["sinfo", "-p", "kshdexclu01", "-h", "-N", "-t", "idle", "-o", "%N"], text=True
).split()
idle_nodes = list(dict.fromkeys(idle_nodes))
if len(idle_nodes) < len(tasks):
    raise SystemExit(f"need {len(tasks)} distinct idle nodes, found {len(idle_nodes)}")
submitted = []
for (route, seed), node in zip(tasks, idle_nodes):
    run_dir = root / "runs" / route / f"seed{seed}"
    if (run_dir / "checkpoints" / "best.pt").exists():
        raise SystemExit(f"refusing to overwrite existing run {run_dir}")
    cmd = ["sbatch", "--parsable", f"--job-name=vcb_{route}_s{seed}", f"--gres=dcu:{allocated_dcu}",
           f"--export=ALL,ALLOCATED_DCU_COUNT={allocated_dcu}", f"--nodelist={node}",
           str(root / "scripts" / "sub_train.slurm"), route, str(seed), str(dcu), str(batch), str(workers)]
    job_id = subprocess.check_output(cmd, text=True).strip()
    submitted.append({"route": route, "seed": seed, "job_id": job_id, "node": node,
                      "allocated_dcu_count": allocated_dcu, "used_dcu_count": dcu,
                      "batch_size_per_rank": batch, "num_workers_total": workers})
manifest = {"resource_selection": selection, "jobs": submitted}
(root / "logs").mkdir(exist_ok=True)
with open(root / "logs" / "formal_jobs.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
