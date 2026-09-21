#!/bin/bash
set -euo pipefail
ROOT=/public/home/putianshu/vis_mlp/visibility_continuous_20260912
mkdir -p ${ROOT}/{configs,models,logs,eval,scripts}
sanity=$(sbatch --parsable ${ROOT}/scripts/sub_sanity.slurm)
r1=$(sbatch --parsable --dependency=afterok:${sanity} --job-name=vc_R1 ${ROOT}/scripts/sub_route.slurm R1)
r2=$(sbatch --parsable --dependency=afterok:${r1} --job-name=vc_R2 ${ROOT}/scripts/sub_route.slurm R2)
r3=$(sbatch --parsable --dependency=afterok:${r2} --job-name=vc_R3 ${ROOT}/scripts/sub_route.slurm R3)
r4=$(sbatch --parsable --dependency=afterok:${r3} --job-name=vc_R4 ${ROOT}/scripts/sub_route.slurm R4)
printf 'stage\tjob_id\tdependency\nR0_sanity\t%s\t\nR1\t%s\t%s\nR2\t%s\t%s\nR3\t%s\t%s\nR4\t%s\t%s\n' "${sanity}" "${r1}" "${sanity}" "${r2}" "${r1}" "${r3}" "${r2}" "${r4}" "${r3}" | tee ${ROOT}/logs/job_chain.tsv
