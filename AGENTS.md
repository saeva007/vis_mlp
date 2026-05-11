# Agent Rules for This Repository

## Git Publishing on This Workspace

- Before committing, inspect `git status --short --branch` and stage only files that belong to the current request.
- Use explicit commit identity if the repository has no local user config:
  `git -c user.name="saeva007" -c user.email="saeva007@gmail.com" commit -m "..."`
- The verified successful push method for this workspace is:
  `git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 push origin main`
- Use that command directly when uploading completed work to `origin/main`, especially after a normal `git push origin main` fails with GitHub HTTPS reset, timeout, RPC disconnect, or `Could not connect to server`.
- In the Codex Desktop PowerShell sandbox, Git operations that write `.git/index.lock`, commit objects, or push over the network may require escalation. Request escalation for the exact Git command instead of trying unrelated workarounds.
- After pushing, verify both local and remote state:
  `git status --short --branch`, `git log --oneline -1`, and, when the GitHub app is available, fetch the pushed commit from `saeva007/vis_mlp`.
- Do not rely on `gh` here unless it has been authenticated in the current environment; `gh auth status` previously reported no logged-in GitHub host.

## Slurm Notes

- Data-build jobs for `.npy/.nc` generation should follow the single-node CPU style used by `sub_s1_data_aerosol_vera.slurm`.
- The airport METAR training script `PMST_net_airport_metar.py` intentionally reuses the original multi-GPU S2/DDP training stack from `PMST_net_test_11_s2_pm10.py`; launch it with `torchrun` through `sub_PMST_net_airport_metar.slurm`.
