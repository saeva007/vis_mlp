#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True, choices=("R5-Gaussian", "R5-Diffusion"))
    parser.add_argument("--stage", required=True, choices=("S1", "S2"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, nargs="+", type=int)
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.route / f"seed{args.seed}" / args.stage
    jobs = []
    short = args.route.split("-")[1][0]
    for step in args.steps:
        checkpoint = run_dir / "checkpoints" / f"ckpt_step_{step:05d}.pt"
        ready = checkpoint.with_suffix(".ready")
        output = run_dir / "validation" / f"full_step_{step:05d}.json"
        if not checkpoint.is_file() or not ready.is_file():
            raise FileNotFoundError(f"checkpoint is not ready: {checkpoint}")
        if output.is_file():
            continue
        command = ["sbatch", "--parsable", f"--job-name=r5snrv_{short}_{args.stage}_s{args.seed}_{step}_f20",
                   str(ROOT / "scripts" / "sub_validate.slurm"), args.route, args.stage, str(args.seed),
                   str(checkpoint), str(output)]
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        jobs.append({"step": step, "job_id": job_id, "checkpoint": str(checkpoint), "output": str(output)})

    if not jobs:
        raise RuntimeError("no validators were submitted")
    dependency = "afterok:" + ":".join(job["job_id"] for job in jobs)
    finalizer_cmd = ["sbatch", "--parsable", f"--dependency={dependency}",
                     f"--job-name=r5snr_finalize_{short}_{args.stage}_retry",
                     str(ROOT / "scripts" / "sub_finalize.slurm"), args.route, args.stage, str(args.seed)]
    finalizer_id = subprocess.check_output(finalizer_cmd, text=True).strip().split(";")[0]
    record = {"route": args.route, "stage": args.stage, "seed": args.seed,
              "screening_reverse_steps": 20, "jobs": jobs,
              "finalizer_job_id": finalizer_id, "dependency": dependency}
    atomic_json(run_dir / "validator_retry_chain.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
