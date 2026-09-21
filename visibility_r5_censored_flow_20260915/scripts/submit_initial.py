#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

ROOT = Path("/public/home/putianshu/vis_mlp/visibility_r5_censored_flow_20260915")
jobs = []
for route in ("R5-CensoredFlow",):
    command = ["sbatch", "--parsable", "--job-name=r5flow_S1_s1",
               str(ROOT / "scripts" / "sub_train.slurm"), route, "S1", "1"]
    jobid = subprocess.check_output(command, text=True).strip().split(";")[0]
    jobs.append({"route": route, "stage": "S1", "seed": 1, "job_id": jobid,
                 "allocated_dcu_count": 4, "training_dcu_count": 4})
(ROOT / "logs" / "initial_jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
print(json.dumps(jobs))
