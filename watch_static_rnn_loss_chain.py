#!/usr/bin/env python3
"""Watch and surgically retry stalled Static-RNN precision-loss jobs.

The watcher manages only job IDs listed in explicit precision-loss manifests.
It never scans or restarts every job owned by a user.  Original manifests stay
immutable; an adjacent ``*_resolved.tsv`` is updated atomically after retries
and must be used for downstream evaluation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import fcntl  # Linux/Slurm; absent during local Windows syntax tests.
except ImportError:  # pragma: no cover - Windows-only fallback
    fcntl = None


REQUIRED_COLUMNS = (
    "candidate_id",
    "candidate_label",
    "seed",
    "stage",
    "run_prefix",
    "run_id",
    "extra_args",
    "s1_job",
    "s2_job",
    "s1_checkpoint",
    "s2_checkpoint",
)

WAITING_STATES = {"PENDING", "CONFIGURING", "RESIZING", "SUSPENDED", "REQUEUED"}
FINALIZING_STATES = {"COMPLETING", "STAGE_OUT"}
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "NODE_FAIL",
    "BOOT_FAIL",
    "PREEMPTED",
    "REVOKED",
    "OUT_OF_MEMORY",
    "TIMEOUT",
    "DEADLINE",
    "SPECIAL_EXIT",
}
RETRY_RECIPE_KEYS = (
    "LOWVIS_RNN_S1_STEPS",
    "LOWVIS_RNN_S2_A_STEPS",
    "LOWVIS_RNN_S2_B_STEPS",
    "LOWVIS_RNN_VAL_INTERVAL",
    "LOWVIS_RNN_BATCH_SIZE",
    "LOWVIS_RNN_GRAD_ACCUM",
    "LOWVIS_RNN_NUM_WORKERS",
    "LOWVIS_RNN_PATIENCE",
)
PROTECTED_RETRY_EXPORTS = {
    "LOWVIS_RNN_MODE",
    "LOWVIS_RNN_EXPERIMENTS",
    "LOWVIS_RNN_RUN_PREFIX",
    "LOWVIS_RNN_RUN_ID",
    "LOWVIS_RNN_LOCAL_CACHE_ID",
    "LOWVIS_RNN_LOCAL_CACHE_DIR",
    "LOWVIS_RNN_CACHE_BASE_DIR",
    "LOWVIS_RNN_CLEAN_LOCAL_CACHE",
    "LOWVIS_RNN_CLEAN_LEGACY_CACHE",
    "LOWVIS_RNN_PRETRAINED_CKPT",
    "LOWVIS_RNN_EXTRA_ARGS",
}
MARKER_RE = re.compile(
    r"(?m)^(?:"
    r"\[(?:S1|S2[^\]]*)\] step=\d+/\d+.*|"
    r"\[(?:S1|S2[^\]]*)\] validation start step=\d+/\d+.*|"
    r"\[(?:S1|S2[^\]]*)\] val score=.*|"
    r"\[(?:S1|S2[^\]]*)\] start steps=.*|"
    r"\[Data-Copy\] (?:Entering RCCL barrier|RCCL barrier passed|Copying|Done:|Cache hit:|Insufficient space|Error:|Warning:).*|"
    r"\[Scaler\] .*|"
    r"\[Data:s[12]\] .*|"
    r"\[Model:S[12]\] .*|"
    r"\[Ckpt\] saved .*|"
    r"\[Dist\] Process group initialized successfully\.|"
    r"Loss experiment .* finished at .*|"
    r"Static-RNN loss-function matrix finished at:.*"
    r")$"
)
STEP_RE = re.compile(r"\[(S1|S2[^\]]*)\] step=(\d+)/(\d+)")
VALIDATION_RE = re.compile(r"\[(S1|S2[^\]]*)\] validation start step=(\d+)/(\d+)")
VAL_SCORE_RE = re.compile(r"\[(S1|S2[^\]]*)\] val score=")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def parse_retry_exports(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in (part.strip() for part in raw.split(";")):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"retry export must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"LOWVIS_RNN_[A-Z0-9_]+", key):
            raise ValueError(f"unsupported retry export name: {key!r}")
        if key in PROTECTED_RETRY_EXPORTS:
            raise ValueError(f"watchdog owns retry export {key}; do not set it in the recipe")
        if not value or any(char in value for char in ",\n\r"):
            raise ValueError(f"unsafe retry export value for {key}: {value!r}")
        if key in values:
            raise ValueError(f"duplicate retry export: {key}")
        values[key] = value
    return values


def parse_only_rows(cli_values: Iterable[str]) -> tuple[str, ...]:
    raw_items = list(cli_values)
    env_value = os.environ.get("WATCH_ONLY_ROWS", "").strip()
    if env_value:
        raw_items.extend(re.split(r"[;\s]+", env_value))
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        value = raw.strip()
        if not value:
            continue
        if not re.fullmatch(r"[^:\s;]+:[^:\s;]+", value):
            raise ValueError(f"managed row must be candidate:seed, got {value!r}")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_job_id(value: str | None) -> str:
    raw = (value or "").strip()
    return raw.split(";", 1)[0].strip()


def normalize_state(value: str | None) -> str:
    raw = (value or "UNKNOWN").strip().upper()
    raw = raw.split("+", 1)[0].split(" ", 1)[0]
    return raw or "UNKNOWN"


def dependency_has_exact_afterok(dependency: str, parent_job_id: str) -> bool:
    parent = normalize_job_id(parent_job_id)
    if not parent:
        return False
    normalized = re.sub(r"\([^)]*\)", "", dependency or "")
    for clause in re.split(r"[?,]", normalized):
        clause = clause.strip()
        if not clause.startswith("afterok:"):
            continue
        job_ids = [normalize_job_id(item) for item in clause[len("afterok:") :].split(":")]
        if parent in job_ids:
            return True
    return False


def run_command(
    command: list[str],
    *,
    check: bool = False,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=f"command timed out after {timeout_seconds}s: {' '.join(command)}",
        )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


@dataclass
class Progress:
    phase: str
    token: str
    marker: str
    marker_offset: int
    step_tag: str | None = None
    step: int | None = None
    total_steps: int | None = None
    log_mtime: float = 0.0


@dataclass
class JobInfo:
    job_id: str
    state: str
    lookup: str = "QUERY_ERROR"
    nodes: str = ""
    reason: str = ""
    stdout: str = ""
    stderr: str = ""
    start_epoch: float = 0.0
    dependency: str = ""
    exit_code: str = ""


@dataclass
class Settings:
    poll_seconds: int
    startup_stale_minutes: int
    data_stale_minutes: int
    train_stale_minutes: int
    validation_stale_minutes: int
    confirmations: int
    max_retries: int
    auto_retry: bool
    exclude_failed_nodes: bool
    once: bool
    recheck_seconds: int
    cancel_wait_seconds: int
    checkpoint_grace_minutes: int
    sbatch_script: str
    local_cache_dir: str
    only_rows: tuple[str, ...]
    retry_exports: dict[str, str]


def read_tail(path: Path, max_bytes: int = 16 * 1024 * 1024) -> tuple[str, int, float]:
    if not path.is_file():
        return "", 0, 0.0
    stat = path.stat()
    start = max(0, stat.st_size - max_bytes)
    with path.open("rb") as handle:
        handle.seek(start)
        payload = handle.read()
    return payload.decode("utf-8", errors="replace"), start, stat.st_mtime


def parse_progress(log_path: Path) -> Progress:
    text, base_offset, mtime = read_tail(log_path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(MARKER_RE.finditer(text))
    marker = matches[-1].group(0).strip() if matches else "<no-semantic-progress>"
    # Noise-only growth must not masquerade as semantic progress when the
    # 16-MiB tail window slides forward.
    offset = base_offset + matches[-1].start() if matches else 0

    steps = list(STEP_RE.finditer(text))
    step_tag: str | None = None
    step: int | None = None
    total: int | None = None
    if steps:
        step_tag = steps[-1].group(1)
        step = int(steps[-1].group(2))
        total = int(steps[-1].group(3))

    validation = list(VALIDATION_RE.finditer(text))
    val_scores = list(VAL_SCORE_RE.finditer(text))
    latest_validation_pos = validation[-1].start() if validation else -1
    latest_val_score_pos = val_scores[-1].start() if val_scores else -1

    if "finished at" in marker:
        phase = "done"
    elif latest_validation_pos > latest_val_score_pos and latest_validation_pos >= 0:
        phase = "validation"
    elif (
        marker.startswith("[Ckpt]")
        and "_latest.pt" in marker
        and step is not None
        and step % 500 == 0
    ):
        # Backward-compatible fallback for logs written before the explicit
        # validation-start heartbeat was added.
        phase = "validation"
    elif marker.startswith("[Data-Copy]") or marker.startswith("[Scaler]"):
        phase = "data"
    elif (
        " step=" in marker
        or " val score=" in marker
        or " start steps=" in marker
        or marker.startswith("[Data:s")
        or marker.startswith("[Model:S")
    ):
        phase = "training"
    else:
        phase = "startup"

    token_payload = f"{phase}|{offset}|{marker}"
    token = hashlib.sha256(token_payload.encode("utf-8", errors="replace")).hexdigest()[:20]
    return Progress(
        phase=phase,
        token=token,
        marker=marker,
        marker_offset=offset,
        step_tag=step_tag,
        step=step,
        total_steps=total,
        log_mtime=mtime,
    )


def checkpoint_signature(row: dict[str, str], stage: str) -> str:
    configured = Path(row[f"{stage}_checkpoint"])
    paths = [configured]
    name = configured.name
    if name.endswith("_best_score.pt"):
        paths.append(configured.with_name(name[: -len("_best_score.pt")] + "_latest.pt"))
    bits: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
            bits.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            bits.append(f"{path.name}:missing")
    return "|".join(bits)


def progress_with_artifacts(log_path: Path, row: dict[str, str], stage: str) -> Progress:
    progress = parse_progress(log_path)
    artifact = checkpoint_signature(row, stage)
    progress.token = hashlib.sha256(f"{progress.token}|{artifact}".encode()).hexdigest()[:20]
    return progress


def parse_scontrol_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S*)", text):
        fields[match.group(1)] = match.group(2)
    return fields


def parse_start_epoch(value: str | None) -> float:
    raw = (value or "").strip()
    if not raw or raw in {"Unknown", "N/A", "None"}:
        return 0.0
    try:
        return dt.datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return 0.0


def query_job(
    job_id: str,
    manifest_dir: Path,
    fallback_job_name: str = "static_rnn_lowvis_loss",
) -> JobInfo:
    job_id = normalize_job_id(job_id)
    if not job_id:
        return JobInfo(job_id="", state="MISSING", lookup="QUERY_ERROR")

    queued = run_command(["squeue", "-h", "-j", job_id, "-o", "%T|%N|%R"])
    lookup = "QUERY_ERROR"
    exit_code = ""
    squeue_reports_absent = (
        queued.returncode == 0 and not queued.stdout.strip()
    ) or (
        queued.returncode != 0
        and "invalid job id" in (queued.stderr or "").lower()
    )
    if queued.returncode == 0 and queued.stdout.strip():
        first = queued.stdout.strip().splitlines()[0]
        parts = first.split("|", 2)
        state = normalize_state(parts[0])
        nodes = "" if len(parts) < 2 or parts[1] in {"(null)", "N/A", "n/a"} else parts[1].strip()
        reason = parts[2].strip() if len(parts) > 2 else ""
        lookup = "LIVE_AUTHORITATIVE"
    elif not squeue_reports_absent:
        state, nodes = "UNKNOWN", ""
        reason = f"squeue query failed rc={queued.returncode}: {queued.stderr.strip()}"
    else:
        state, nodes, reason = "UNKNOWN", "", ""
        accounting = run_command(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--format=JobIDRaw,State,NodeList,ExitCode",
            ]
        )
        if accounting.returncode == 0:
            for line in accounting.stdout.splitlines():
                if not line.strip():
                    continue
                fields = line.split("|")
                if not fields or normalize_job_id(fields[0]) != job_id:
                    continue
                candidate_state = normalize_state(fields[1] if len(fields) > 1 else "UNKNOWN")
                nodes = fields[2].strip() if len(fields) > 2 else ""
                exit_code = fields[3].strip() if len(fields) > 3 else ""
                reason = f"ExitCode={exit_code}" if exit_code else ""
                if candidate_state in TERMINAL_STATES:
                    state = candidate_state
                    lookup = "TERMINAL_AUTHORITATIVE"
                else:
                    state = "UNKNOWN"
                    lookup = "ACCOUNTING_PENDING"
                    reason = f"squeue empty but sacct state={candidate_state}"
                break
            else:
                lookup = "ACCOUNTING_PENDING"
                reason = "squeue empty and exact allocation row is not yet visible in sacct"
        else:
            lookup = "QUERY_ERROR"
            reason = f"sacct query failed rc={accounting.returncode}: {accounting.stderr.strip()}"

    control = run_command(["scontrol", "show", "job", "-o", job_id])
    control_fields = parse_scontrol_fields(control.stdout) if control.returncode == 0 else {}
    stdout_path = control_fields.get("StdOut", "")
    stderr_path = control_fields.get("StdErr", "")
    if stdout_path in {"(null)", "N/A", "n/a"}:
        stdout_path = ""
    if stderr_path in {"(null)", "N/A", "n/a"}:
        stderr_path = ""
    job_name = control_fields.get("JobName", fallback_job_name)
    for token, replacement in (("%j", job_id), ("%A", job_id), ("%x", job_name)):
        stdout_path = stdout_path.replace(token, replacement)
        stderr_path = stderr_path.replace(token, replacement)
    if not stdout_path:
        stdout_path = str(manifest_dir / f"{job_id}_{job_name}.out")
    if not stderr_path:
        stderr_path = str(manifest_dir / f"{job_id}_{job_name}.err")
    if not nodes:
        nodes = control_fields.get("NodeList", "")
        if nodes in {"(null)", "N/A", "n/a"}:
            nodes = ""
    return JobInfo(
        job_id=job_id,
        state=state,
        lookup=lookup,
        nodes=nodes,
        reason=reason or control_fields.get("Reason", ""),
        stdout=stdout_path,
        stderr=stderr_path,
        start_epoch=parse_start_epoch(control_fields.get("StartTime")),
        dependency=control_fields.get("Dependency", ""),
        exit_code=exit_code,
    )


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"manifest missing required columns {missing}: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"manifest has no candidate rows: {path}")
    return fieldnames, rows


def atomic_write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "rows": {}, "jobs": {}, "blocked": {}}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("version", 1)
    state.setdefault("rows", {})
    state.setdefault("jobs", {})
    state.setdefault("blocked", {})
    return state


def action_header() -> list[str]:
    return [
        "time",
        "candidate_id",
        "seed",
        "stage",
        "action",
        "old_job",
        "new_job",
        "nodes",
        "reason",
        "run_prefix",
        "cache_id",
    ]


def append_action(path: Path, **values: str) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=action_header(), delimiter="\t", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({name: values.get(name, "") for name in action_header()})
        handle.flush()
        os.fsync(handle.fileno())


def phase_threshold_minutes(progress: Progress, settings: Settings) -> int:
    if progress.phase == "validation":
        return settings.validation_stale_minutes
    if progress.phase == "data":
        return settings.data_stale_minutes
    if progress.phase == "training":
        return settings.train_stale_minutes
    return settings.startup_stale_minutes


def base_retry_prefix(run_prefix: str) -> str:
    return re.sub(r"_wd_r\d+_\d{8}T\d{6}$", "", run_prefix)


def retry_job_name(row: dict[str, str], stage: str, attempt: int) -> str:
    return f"lv_{row['candidate_id']}_s{row['seed']}_{stage}_r{attempt}"[:128]


def build_export(
    *,
    mode: str,
    run_prefix: str,
    cache_id: str,
    settings: Settings,
    retry_exports: dict[str, str],
    pretrained_ckpt: str = "",
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_id):
        raise ValueError(f"unsafe retry cache id: {cache_id!r}")
    cache_base = settings.local_cache_dir.rstrip("/")
    if not cache_base or cache_base == "/":
        raise ValueError(f"unsafe WATCH_LOCAL_CACHE_DIR: {settings.local_cache_dir!r}")
    scoped_cache_dir = f"{cache_base}/{cache_id}"
    values = dict(retry_exports)
    values.update({
        "LOWVIS_RNN_MODE": mode,
        "LOWVIS_RNN_EXPERIMENTS": "2",
        "LOWVIS_RNN_RUN_PREFIX": run_prefix,
        "LOWVIS_RNN_LOCAL_CACHE_ID": cache_id,
        "LOWVIS_RNN_CLEAN_LOCAL_CACHE": "1",
        "LOWVIS_RNN_CLEAN_LEGACY_CACHE": "1",
        "LOWVIS_RNN_LOCAL_CACHE_DIR": scoped_cache_dir,
        "LOWVIS_RNN_CACHE_BASE_DIR": cache_base,
    })
    if pretrained_ckpt:
        values["LOWVIS_RNN_PRETRAINED_CKPT"] = pretrained_ckpt
    for key, value in values.items():
        if "," in value or "\n" in value:
            raise ValueError(f"unsafe comma/newline in sbatch export {key}={value!r}")
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def submit_stage(
    *,
    row: dict[str, str],
    stage: str,
    run_prefix: str,
    cache_id: str,
    settings: Settings,
    attempt: int,
    exclude_nodes: str,
    retry_exports: dict[str, str],
    decision_id: str,
    dependency: str = "",
    pretrained_ckpt: str = "",
) -> str:
    job_name = retry_job_name(row, stage, attempt)
    comment = f"lowvis-watch:{decision_id}:{stage}"
    command = ["sbatch", "--parsable", f"--job-name={job_name}", f"--comment={comment}"]
    if exclude_nodes and settings.exclude_failed_nodes:
        command.append(f"--exclude={exclude_nodes}")
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(
        "--export="
        + build_export(
            mode=stage,
            run_prefix=run_prefix,
            cache_id=cache_id,
            settings=settings,
            retry_exports=retry_exports,
            pretrained_ckpt=pretrained_ckpt,
        )
    )
    command.append(settings.sbatch_script)
    submit_env = os.environ.copy()
    submit_env["LOWVIS_RNN_EXTRA_ARGS"] = row["extra_args"]
    result = run_command(command, check=True, env=submit_env)
    job_id = normalize_job_id(result.stdout.strip().splitlines()[-1])
    if not job_id or not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
        raise RuntimeError(f"sbatch returned an invalid job id: {result.stdout!r}")
    return job_id


class ManifestWatch:
    def __init__(self, original: Path, settings: Settings, custom_resolved: str = "") -> None:
        self.original = original.resolve()
        if not self.original.is_file():
            raise FileNotFoundError(f"watch manifest not found: {self.original}")
        if custom_resolved:
            self.resolved = Path(custom_resolved).resolve()
        else:
            self.resolved = self.original.with_name(f"{self.original.stem}_resolved.tsv")
        if self.resolved == self.original:
            raise ValueError("resolved manifest must differ from the immutable original manifest")
        self.state_path = self.original.with_name(f"{self.original.stem}_watchdog_state.json")
        self.action_path = self.original.with_name(f"{self.original.stem}_watchdog_actions.tsv")
        self.lock_path = self.original.with_name(f"{self.original.stem}_watchdog.lock")
        self.settings = settings
        self.lock_handle: Any = None

        self.acquire_lock()
        try:
            if not self.resolved.exists():
                shutil.copy2(self.original, self.resolved)
            self.fieldnames, self.rows = read_manifest(self.resolved)
            self.state = load_state(self.state_path)
            original_fingerprint = file_sha256(self.original)
            recorded_fingerprint = self.state.get("original_sha256", "")
            if recorded_fingerprint and recorded_fingerprint != original_fingerprint:
                raise RuntimeError(
                    f"original manifest changed after watchdog state was created: {self.original}"
                )
            self.state["original_sha256"] = original_fingerprint
            allowed = set(self.settings.only_rows)
            self.managed_rows = [
                row
                for row in self.rows
                if not allowed or f"{row['candidate_id']}:{row['seed']}" in allowed
            ]
            if not self.managed_rows:
                raise ValueError(f"manifest has no explicitly managed rows: {self.original}")
            logical_keys = [self.row_key(row) for row in self.managed_rows]
            if len(logical_keys) != len(set(logical_keys)):
                raise ValueError(f"duplicate candidate/seed rows in manifest: {self.original}")
            if self.settings.auto_retry:
                for row in self.managed_rows:
                    self.retry_recipe(row)
            unfinished = {
                key: value.get("submit_intent")
                for key, value in self.state.get("rows", {}).items()
                if value.get("submit_intent")
            }
            if unfinished:
                raise RuntimeError(
                    "unfinished watchdog submit intent detected; refusing to risk duplicate jobs: "
                    + json.dumps(unfinished, ensure_ascii=False, sort_keys=True)
                )
            self.save()
        except Exception:
            self.close()
            raise

    def acquire_lock(self) -> None:
        self.lock_handle = self.lock_path.open("a+", encoding="utf-8")
        if fcntl is not None:
            try:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another watchdog owns {self.lock_path}") from exc
        self.lock_handle.seek(0)
        self.lock_handle.truncate()
        self.lock_handle.write(f"pid={os.getpid()} slurm_job={os.environ.get('SLURM_JOB_ID', '')}\n")
        self.lock_handle.flush()

    def close(self) -> None:
        if self.lock_handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_handle.close()
            self.lock_handle = None

    def save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def row_key(self, row: dict[str, str]) -> str:
        return f"{row['candidate_id']}|{row['seed']}"

    def retry_recipe(self, row: dict[str, str]) -> dict[str, str]:
        raw = row.get("retry_exports", "").strip()
        recipe = parse_retry_exports(raw) if raw else dict(self.settings.retry_exports)
        missing = [key for key in RETRY_RECIPE_KEYS if key not in recipe]
        if missing:
            raise ValueError(
                f"missing retry recipe values for {self.row_key(row)}: {missing}; "
                "use a manifest with retry_exports or set WATCH_RETRY_EXPORTS"
            )
        return recipe

    def query(self, row: dict[str, str], stage: str, job_id: str) -> JobInfo:
        job_state = self.state.get("jobs", {}).get(normalize_job_id(job_id), {})
        fallback_name = job_state.get("job_name", "static_rnn_lowvis_loss")
        return query_job(job_id, self.resolved.parent, fallback_name)

    def log_action(self, row: dict[str, str], stage: str, action: str, **kwargs: str) -> None:
        append_action(
            self.action_path,
            time=now_iso(),
            candidate_id=row["candidate_id"],
            seed=row["seed"],
            stage=stage,
            action=action,
            old_job=kwargs.get("old_job", ""),
            new_job=kwargs.get("new_job", ""),
            nodes=kwargs.get("nodes", ""),
            reason=kwargs.get("reason", ""),
            run_prefix=kwargs.get("run_prefix", row["run_prefix"]),
            cache_id=kwargs.get("cache_id", ""),
        )

    def observe(self, row: dict[str, str], stage: str, info: JobInfo) -> tuple[Progress, float, int]:
        progress = progress_with_artifacts(Path(info.stdout), row, stage)
        job_state = self.state["jobs"].setdefault(info.job_id, {})
        current_time = time.time()
        if job_state.get("token") != progress.token:
            initial_time = current_time
            if "token" not in job_state:
                evidence_time = max(progress.log_mtime, info.start_epoch)
                if evidence_time > 0:
                    initial_time = min(current_time, evidence_time)
            job_state.update(
                {
                    "token": progress.token,
                    "last_change_epoch": initial_time,
                    "confirmations": 0,
                    "phase": progress.phase,
                    "marker": progress.marker,
                    "last_seen": now_iso(),
                }
            )
        else:
            job_state["last_seen"] = now_iso()

        age_seconds = max(0.0, current_time - float(job_state.get("last_change_epoch", current_time)))
        threshold = phase_threshold_minutes(progress, self.settings) * 60
        if age_seconds >= threshold:
            job_state["confirmations"] = int(job_state.get("confirmations", 0)) + 1
        else:
            job_state["confirmations"] = 0
        return progress, age_seconds, int(job_state["confirmations"])

    def recheck_stale_running(
        self,
        row: dict[str, str],
        stage: str,
        info: JobInfo,
        old_token: str,
    ) -> JobInfo | None:
        if self.settings.recheck_seconds > 0:
            time.sleep(self.settings.recheck_seconds)
        refreshed = self.query(row, stage, info.job_id)
        if refreshed.lookup != "LIVE_AUTHORITATIVE" or refreshed.state != "RUNNING":
            detail = f"lookup={refreshed.lookup} state={refreshed.state} reason={refreshed.reason}"
            self.log_action(row, stage, "ABORT_RECHECK_STATE", old_job=info.job_id, reason=detail)
            print(f"[watchdog] recovery deferred for {info.job_id}: {detail}", flush=True)
            return None
        if (
            info.start_epoch > 0
            and refreshed.start_epoch > 0
            and abs(info.start_epoch - refreshed.start_epoch) > 1
        ):
            detail = (
                f"Slurm start time changed from {info.start_epoch} to {refreshed.start_epoch}; "
                "job may have been requeued"
            )
            self.log_action(row, stage, "ABORT_RECHECK_ATTEMPT", old_job=info.job_id, reason=detail)
            print(f"[watchdog] recovery deferred for {info.job_id}: {detail}", flush=True)
            return None
        new_progress = progress_with_artifacts(Path(refreshed.stdout), row, stage)
        if new_progress.token != old_token:
            job_state = self.state["jobs"].setdefault(info.job_id, {})
            job_state["token"] = new_progress.token
            job_state["confirmations"] = 0
            job_state["last_change_epoch"] = time.time()
            self.log_action(
                row,
                stage,
                "RECHECK_PROGRESS",
                old_job=info.job_id,
                reason="new semantic progress before cancel",
            )
            print(f"[watchdog] progress resumed; keeping {info.job_id}", flush=True)
            return None
        return refreshed

    def derive_retry_paths(
        self,
        row: dict[str, str],
        stage: str,
        retry_prefix: str,
    ) -> tuple[str, str, str]:
        old_prefix = row["run_prefix"]
        old_run_id = row["run_id"]
        if not old_run_id.startswith(old_prefix):
            raise ValueError(
                f"run_id does not start with run_prefix for {self.row_key(row)}: "
                f"{old_run_id!r} vs {old_prefix!r}"
            )
        run_id = retry_prefix + old_run_id[len(old_prefix) :]

        def replace_checkpoint(raw: str, label: str) -> str:
            path = Path(raw)
            if old_run_id not in path.name:
                raise ValueError(
                    f"{label} checkpoint name does not contain run_id for {self.row_key(row)}: {raw}"
                )
            return str(path.with_name(path.name.replace(old_run_id, run_id, 1)))

        s1_checkpoint = row["s1_checkpoint"]
        if stage == "s1":
            s1_checkpoint = replace_checkpoint(s1_checkpoint, "S1")
        return (run_id, s1_checkpoint, replace_checkpoint(row["s2_checkpoint"], "S2"))

    def cancel_stale_attempt(
        self,
        row: dict[str, str],
        stage: str,
        info: JobInfo,
        token: str,
        intent: dict[str, Any],
    ) -> bool:
        # Third authoritative snapshot immediately before scancel.  Any state
        # transition or query uncertainty aborts recovery without mutation.
        primary = self.query(row, stage, info.job_id)
        if primary.lookup != "LIVE_AUTHORITATIVE" or primary.state != "RUNNING":
            detail = f"lookup={primary.lookup} state={primary.state} reason={primary.reason}"
            self.log_action(row, stage, "ABORT_PRE_CANCEL_STATE", old_job=info.job_id, reason=detail)
            print(f"[watchdog] recovery deferred for {info.job_id}: {detail}", flush=True)
            return False
        if (
            info.start_epoch > 0
            and primary.start_epoch > 0
            and abs(info.start_epoch - primary.start_epoch) > 1
        ):
            detail = (
                f"Slurm start time changed from {info.start_epoch} to {primary.start_epoch}; "
                "refusing to cancel a requeued attempt"
            )
            self.log_action(row, stage, "ABORT_PRE_CANCEL_ATTEMPT", old_job=info.job_id, reason=detail)
            print(f"[watchdog] recovery deferred for {info.job_id}: {detail}", flush=True)
            return False
        latest = progress_with_artifacts(Path(primary.stdout), row, stage)
        if latest.token != token:
            self.log_action(
                row,
                stage,
                "ABORT_PRE_CANCEL_PROGRESS",
                old_job=info.job_id,
                reason="new semantic progress immediately before cancel",
            )
            print(f"[watchdog] progress resumed; keeping {info.job_id}", flush=True)
            return False

        targets: list[tuple[str, str]] = [(info.job_id, stage)]
        if stage == "s1" and normalize_job_id(row.get("s2_job")):
            dependent_id = normalize_job_id(row["s2_job"])
            dependent = self.query(row, "s2", dependent_id)
            if dependent.lookup == "LIVE_AUTHORITATIVE":
                if dependent.state not in WAITING_STATES:
                    detail = (
                        f"dependent S2 {dependent_id} is {dependent.state}, not safely pending; "
                        "manifest/dependency requires manual review"
                    )
                    self.log_action(row, stage, "ABORT_DEPENDENT_STATE", old_job=info.job_id, reason=detail)
                    print(f"[watchdog] {detail}", flush=True)
                    return False
                if not dependency_has_exact_afterok(dependent.dependency, info.job_id):
                    detail = (
                        f"dependent S2 {dependent_id} does not report afterok:{info.job_id}; "
                        f"Dependency={dependent.dependency!r}"
                    )
                    self.log_action(row, stage, "ABORT_DEPENDENCY_MISMATCH", old_job=info.job_id, reason=detail)
                    print(f"[watchdog] {detail}", flush=True)
                    return False
                targets.append((dependent_id, "s2"))
            elif not (
                dependent.lookup == "TERMINAL_AUTHORITATIVE" and dependent.state == "CANCELLED"
            ):
                detail = (
                    f"dependent S2 {dependent_id} lookup={dependent.lookup} "
                    f"state={dependent.state}; recovery deferred"
                )
                self.log_action(row, stage, "ABORT_DEPENDENT_QUERY", old_job=info.job_id, reason=detail)
                print(f"[watchdog] {detail}", flush=True)
                return False

        row_state = self.state["rows"].setdefault(self.row_key(row), {})
        intent.update(
            {
                "phase": "PRE_CANCEL",
                "old_jobs": [job_id for job_id, _ in targets],
                "created": now_iso(),
            }
        )
        row_state["submit_intent"] = intent
        self.save()

        def abort_persisted_intent(detail: str) -> bool:
            aborted = dict(intent)
            aborted.update({"phase": "ABORTED_BEFORE_SCANCEL", "aborted": now_iso(), "detail": detail})
            row_state["last_aborted_intent"] = aborted
            row_state.pop("submit_intent", None)
            self.log_action(
                row,
                stage,
                "ABORT_POST_INTENT_RECHECK",
                old_job=info.job_id,
                reason=detail,
            )
            self.save()
            print(f"[watchdog] recovery deferred for {info.job_id}: {detail}", flush=True)
            return False

        # The fsync above can be slow on NFS.  Revalidate after it so that the
        # last query-to-scancel window contains no state-file I/O.
        final_primary = self.query(row, stage, info.job_id)
        if final_primary.lookup != "LIVE_AUTHORITATIVE" or final_primary.state != "RUNNING":
            return abort_persisted_intent(
                f"post-intent primary lookup={final_primary.lookup} state={final_primary.state}"
            )
        if (
            info.start_epoch > 0
            and final_primary.start_epoch > 0
            and abs(info.start_epoch - final_primary.start_epoch) > 1
        ):
            return abort_persisted_intent("post-intent Slurm start time changed; possible requeue")
        final_progress = progress_with_artifacts(Path(final_primary.stdout), row, stage)
        if final_progress.token != token:
            return abort_persisted_intent("semantic progress resumed after intent was persisted")
        if stage == "s1" and len(targets) > 1:
            dependent_id = targets[1][0]
            final_dependent = self.query(row, "s2", dependent_id)
            if (
                final_dependent.lookup != "LIVE_AUTHORITATIVE"
                or final_dependent.state not in WAITING_STATES
                or not dependency_has_exact_afterok(final_dependent.dependency, info.job_id)
            ):
                return abort_persisted_intent(
                    f"post-intent S2 {dependent_id} lookup={final_dependent.lookup} "
                    f"state={final_dependent.state} Dependency={final_dependent.dependency!r}"
                )

        cancel_ids = [job_id for job_id, _ in targets]
        cancelled = run_command(["scancel", *cancel_ids])
        if cancelled.returncode != 0:
            intent["phase"] = "CANCEL_COMMAND_UNCERTAIN"
            intent["error"] = cancelled.stderr.strip() or cancelled.stdout.strip()
            self.save()
            raise RuntimeError(
                f"scancel failed for {cancel_ids}; refusing to submit replacements: {intent['error']}"
            )
        intent["phase"] = "CANCEL_REQUESTED"
        self.save()

        deadline = time.time() + self.settings.cancel_wait_seconds
        remaining = {job_id: target_stage for job_id, target_stage in targets}
        while remaining and time.time() < deadline:
            for job_id, target_stage in list(remaining.items()):
                status = self.query(row, target_stage, job_id)
                if status.lookup == "TERMINAL_AUTHORITATIVE":
                    if status.state != "CANCELLED":
                        intent["phase"] = "CANCEL_RACE_TERMINAL"
                        intent["terminal_state"] = status.state
                        self.save()
                        raise RuntimeError(
                            f"job {job_id} became {status.state}, not CANCELLED; "
                            "refusing to submit a replacement"
                        )
                    remaining.pop(job_id)
            if remaining:
                time.sleep(2)
        if remaining:
            intent["phase"] = "CANCEL_CONFIRMATION_TIMEOUT"
            intent["remaining"] = sorted(remaining)
            self.save()
            raise RuntimeError(
                f"could not confirm CANCELLED in sacct for {sorted(remaining)}; "
                "refusing to submit replacements"
            )
        intent["phase"] = "CANCEL_CONFIRMED"
        self.save()
        return True

    def cleanup_submitted_replacements(
        self,
        row: dict[str, str],
        attempt: int,
        decision_id: str,
        known_jobs: dict[str, str],
    ) -> list[str]:
        jobs = {
            normalize_job_id(job_id): stage
            for job_id, stage in known_jobs.items()
            if normalize_job_id(job_id)
        }
        discovered = run_command(["squeue", "-h", "--me", "-o", "%i|%k"])
        if discovered.returncode == 0:
            for line in discovered.stdout.splitlines():
                fields = line.strip().split("|", 1)
                if len(fields) != 2:
                    continue
                job_id, comment = normalize_job_id(fields[0]), fields[1].strip()
                match = re.fullmatch(rf"lowvis-watch:{re.escape(decision_id)}:(s1|s2)", comment)
                if job_id and re.fullmatch(r"\d+(?:_[0-9]+)?", job_id) and match:
                    jobs.setdefault(job_id, match.group(1))

        problems: list[str] = []
        ordered = sorted(jobs.items(), key=lambda item: 0 if item[1] == "s2" else 1)
        for job_id, target_stage in ordered:
            cancelled = run_command(["scancel", job_id])
            if cancelled.returncode != 0:
                problems.append(
                    f"scancel {job_id} failed: {cancelled.stderr.strip() or cancelled.stdout.strip()}"
                )
                continue
            deadline = time.time() + self.settings.cancel_wait_seconds
            fallback_name = retry_job_name(row, target_stage, attempt)
            while time.time() < deadline:
                status = query_job(job_id, self.resolved.parent, fallback_name)
                if status.lookup == "TERMINAL_AUTHORITATIVE":
                    if status.state != "CANCELLED":
                        problems.append(f"replacement {job_id} became {status.state}, not CANCELLED")
                    break
                time.sleep(2)
            else:
                problems.append(f"replacement {job_id} cancellation was not confirmed")
        if not jobs:
            problems.append(
                f"no replacement JobID could be recovered for decision_id={decision_id}; "
                "inspect Slurm comments manually"
            )
        return problems

    def restart(self, row: dict[str, str], stage: str, info: JobInfo, reason: str, token: str) -> bool:
        key = self.row_key(row)
        row_state = self.state["rows"].setdefault(key, {})
        if not self.settings.auto_retry:
            report_key = f"reported|{stage}|{info.job_id}|{token}"
            if row_state.get("last_report") != report_key:
                row_state["last_report"] = report_key
                self.log_action(row, stage, "SUSPECT", old_job=info.job_id, nodes=info.nodes, reason=reason)
                print(f"[watchdog] SUSPECT {key} {stage} job={info.job_id}: {reason}", flush=True)
            return False

        confirmed = self.recheck_stale_running(row, stage, info, token)
        if confirmed is None:
            return False

        retries = int(row_state.get("retry_count", 0))
        attempt = retries + 1
        decision_id = uuid.uuid4().hex[:16]
        base_intent: dict[str, Any] = {
            "decision_id": decision_id,
            "attempt": attempt,
            "stage": stage,
            "reason": reason,
        }
        if retries >= self.settings.max_retries:
            base_intent["cancel_only"] = True
            if self.cancel_stale_attempt(row, stage, confirmed, token, base_intent):
                row_state.pop("submit_intent", None)
                message = f"retry limit reached ({retries}/{self.settings.max_retries}): {reason}"
                self.state["blocked"][f"{key}|{stage}"] = message
                self.log_action(
                    row,
                    stage,
                    "CANCELLED_AT_RETRY_LIMIT",
                    old_job=info.job_id,
                    nodes=confirmed.nodes,
                    reason=message,
                )
                print(f"[watchdog] BLOCKED {key} {stage}: {message}", flush=True)
                self.save()
            return False

        stamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        retry_prefix = f"{base_retry_prefix(row['run_prefix'])}_wd_r{attempt}_{stamp}"
        retry_cache = f"{retry_prefix}_cache"
        run_id, s1_ckpt, s2_ckpt = self.derive_retry_paths(row, stage, retry_prefix)
        recipe = self.retry_recipe(row)
        if not Path(row["s1_checkpoint"]).parent.is_dir():
            raise FileNotFoundError(
                f"checkpoint directory is missing: {Path(row['s1_checkpoint']).parent}"
            )
        pretrained = row["s1_checkpoint"]
        if stage == "s2":
            pretrained_path = Path(pretrained)
            if not pretrained_path.is_file() or pretrained_path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"non-empty S1 checkpoint required for S2 retry is missing: {pretrained}"
                )
        failed_node_lists = list(row_state.get("failed_node_lists", []))
        if self.settings.exclude_failed_nodes and confirmed.nodes and confirmed.nodes not in failed_node_lists:
            failed_node_lists.append(confirmed.nodes)
        exclude_nodes = ",".join(failed_node_lists) if self.settings.exclude_failed_nodes else ""
        if exclude_nodes and not re.fullmatch(r"[A-Za-z0-9_,.\-\[\]]+", exclude_nodes):
            raise ValueError(f"unsafe Slurm NodeList for --exclude: {exclude_nodes!r}")

        # Complete all local validation before cancelling the old allocation.
        build_export(
            mode="s1" if stage == "s1" else "s2",
            run_prefix=retry_prefix,
            cache_id=retry_cache,
            settings=self.settings,
            retry_exports=recipe,
            pretrained_ckpt="" if stage == "s1" else pretrained,
        )
        if stage == "s1":
            build_export(
                mode="s2",
                run_prefix=retry_prefix,
                cache_id=retry_cache,
                settings=self.settings,
                retry_exports=recipe,
                pretrained_ckpt=s1_ckpt,
            )

        base_intent.update(
            {
                "retry_prefix": retry_prefix,
                "retry_cache": retry_cache,
                "run_id": run_id,
                "exclude_nodes": exclude_nodes,
            }
        )
        if not self.cancel_stale_attempt(row, stage, confirmed, token, base_intent):
            return False

        new_s1_job = row["s1_job"]
        new_s2_job = ""
        try:
            if stage == "s1":
                new_s1_job = submit_stage(
                    row=row,
                    stage="s1",
                    run_prefix=retry_prefix,
                    cache_id=retry_cache,
                    settings=self.settings,
                    attempt=attempt,
                    exclude_nodes=exclude_nodes,
                    retry_exports=recipe,
                    decision_id=decision_id,
                )
                row_state["submit_intent"].update(
                    {"phase": "S1_SUBMITTED", "new_s1_job": new_s1_job}
                )
                self.state["jobs"].setdefault(new_s1_job, {})["job_name"] = retry_job_name(
                    row, "s1", attempt
                )
                self.save()
                new_s2_job = submit_stage(
                    row=row,
                    stage="s2",
                    run_prefix=retry_prefix,
                    cache_id=retry_cache,
                    settings=self.settings,
                    attempt=attempt,
                    exclude_nodes=exclude_nodes,
                    retry_exports=recipe,
                    decision_id=decision_id,
                    dependency=new_s1_job,
                    pretrained_ckpt=s1_ckpt,
                )
            else:
                new_s2_job = submit_stage(
                    row=row,
                    stage="s2",
                    run_prefix=retry_prefix,
                    cache_id=retry_cache,
                    settings=self.settings,
                    attempt=attempt,
                    exclude_nodes=exclude_nodes,
                    retry_exports=recipe,
                    decision_id=decision_id,
                    pretrained_ckpt=pretrained,
                )
            row_state["submit_intent"].update(
                {"phase": "S2_SUBMITTED", "new_s2_job": new_s2_job}
            )
            self.state["jobs"].setdefault(new_s2_job, {})["job_name"] = retry_job_name(
                row, "s2", attempt
            )
            self.save()
        except Exception:
            known_jobs: dict[str, str] = {}
            if normalize_job_id(new_s2_job) and new_s2_job != row["s2_job"]:
                known_jobs[new_s2_job] = "s2"
            if stage == "s1" and normalize_job_id(new_s1_job) and new_s1_job != row["s1_job"]:
                known_jobs[new_s1_job] = "s1"
            cleanup_problems = self.cleanup_submitted_replacements(
                row,
                attempt,
                decision_id,
                known_jobs,
            )
            row_state["submit_intent"]["failed"] = now_iso()
            row_state["submit_intent"]["replacement_cleanup_problems"] = cleanup_problems
            try:
                self.save()
            except Exception as save_error:
                print(
                    f"[watchdog] could not persist failed-intent cleanup: {save_error}",
                    file=sys.stderr,
                    flush=True,
                )
            if cleanup_problems:
                print(
                    f"[watchdog] replacement cleanup requires manual audit: {cleanup_problems}",
                    file=sys.stderr,
                    flush=True,
                )
            raise

        row["run_prefix"] = retry_prefix
        row["run_id"] = run_id
        row["s2_job"] = new_s2_job
        row["s2_checkpoint"] = s2_ckpt
        if stage == "s1":
            row["s1_job"] = new_s1_job
            row["s1_checkpoint"] = s1_ckpt
        row_state["retry_count"] = attempt
        row_state["failed_node_lists"] = failed_node_lists
        row_state["last_retry"] = now_iso()
        row_state["last_report"] = ""
        row_state.pop("submit_intent", None)
        self.state["blocked"].pop(f"{key}|{stage}", None)
        atomic_write_manifest(self.resolved, self.fieldnames, self.rows)
        self.save()
        new_job_display = f"{new_s1_job}->{new_s2_job}" if stage == "s1" else new_s2_job
        self.log_action(
            row,
            stage,
            "RETRY_SUBMITTED",
            old_job=info.job_id,
            new_job=new_job_display,
            nodes=exclude_nodes,
            reason=reason,
            run_prefix=retry_prefix,
            cache_id=retry_cache,
        )
        print(
            f"[watchdog] RETRY {key} {stage}: {info.job_id} -> {new_job_display}; "
            f"exclude={exclude_nodes or '<none>'}",
            flush=True,
        )
        return True

    def process_job(self, row: dict[str, str], stage: str, info: JobInfo) -> str:
        key = self.row_key(row)
        if info.lookup not in {"LIVE_AUTHORITATIVE", "TERMINAL_AUTHORITATIVE"}:
            print(
                f"[watchdog] {key} {stage} job={info.job_id} lookup={info.lookup} "
                f"state={info.state} reason={info.reason}; no action",
                flush=True,
            )
            return "active"

        if info.state == "RUNNING":
            progress, age_seconds, confirmations = self.observe(row, stage, info)
            threshold = phase_threshold_minutes(progress, self.settings)
            print(
                f"[watchdog] {key} {stage} job={info.job_id} RUNNING phase={progress.phase} "
                f"age={age_seconds / 60:.1f}m threshold={threshold}m "
                f"confirm={confirmations}/{self.settings.confirmations} marker={progress.marker[:180]}",
                flush=True,
            )
            if confirmations >= self.settings.confirmations:
                reason = (
                    f"semantic progress unchanged for {age_seconds / 60:.1f} min "
                    f"in phase={progress.phase}; marker={progress.marker[:300]}"
                )
                if self.restart(row, stage, info, reason, progress.token):
                    return "restarted"
            return "active"

        if info.state in WAITING_STATES | FINALIZING_STATES:
            print(
                f"[watchdog] {key} {stage} job={info.job_id} {info.state} reason={info.reason}",
                flush=True,
            )
            return "active"

        if info.state == "COMPLETED":
            return "completed"

        if info.lookup == "TERMINAL_AUTHORITATIVE":
            message = (
                f"terminal state={info.state} reason={info.reason}; automatic recovery is limited "
                "to authoritatively confirmed stale RUNNING jobs"
            )
            self.state["blocked"][f"{key}|{stage}"] = message
            row_state = self.state["rows"].setdefault(key, {})
            report_key = f"terminal|{stage}|{info.job_id}|{info.state}"
            if row_state.get("last_report") != report_key:
                row_state["last_report"] = report_key
                self.log_action(row, stage, "BLOCKED", old_job=info.job_id, nodes=info.nodes, reason=message)
                print(f"[watchdog] BLOCKED {key} {stage}: {message}", flush=True)
            return "blocked"

        print(f"[watchdog] {key} {stage} job={info.job_id} state={info.state}; waiting", flush=True)
        return "active"

    def completed_artifact_status(
        self,
        row: dict[str, str],
        stage: str,
        info: JobInfo,
    ) -> str:
        key = self.row_key(row)
        path = Path(row[f"{stage}_checkpoint"])
        valid = False
        problem = "missing"
        try:
            stat = path.stat()
            if stat.st_size <= 0:
                problem = "empty"
            elif info.start_epoch > 0 and stat.st_mtime < info.start_epoch - 300:
                problem = "older than the Slurm job start"
            else:
                valid = True
        except OSError:
            pass

        row_state = self.state["rows"].setdefault(key, {})
        wait_key = f"artifact_wait|{stage}|{info.job_id}"
        blocked_key = f"{key}|{stage}"
        if valid:
            row_state.pop(wait_key, None)
            if self.state["blocked"].get(blocked_key, "").startswith("completed job has"):
                self.state["blocked"].pop(blocked_key, None)
            return "completed"

        current_time = time.time()
        first_seen = float(row_state.setdefault(wait_key, current_time))
        age_minutes = max(0.0, current_time - first_seen) / 60
        if age_minutes < self.settings.checkpoint_grace_minutes:
            print(
                f"[watchdog] {key} {stage} completed; checkpoint {problem}: {path}; "
                f"artifact grace={age_minutes:.1f}/{self.settings.checkpoint_grace_minutes}m",
                flush=True,
            )
            return "active"

        message = f"completed job has {problem} checkpoint after grace period: {path}"
        self.state["blocked"][blocked_key] = message
        report_key = f"artifact|{stage}|{info.job_id}|{problem}"
        if row_state.get("last_report") != report_key:
            row_state["last_report"] = report_key
            self.log_action(row, stage, "BLOCKED_ARTIFACT", old_job=info.job_id, reason=message)
            print(f"[watchdog] BLOCKED {key} {stage}: {message}", flush=True)
        return "blocked"

    def quiesce_downstream(self, row: dict[str, str], parent_reason: str) -> bool:
        dependent_id = normalize_job_id(row.get("s2_job"))
        if not dependent_id:
            return True
        parent_id = normalize_job_id(row.get("s1_job"))
        dependent = self.query(row, "s2", dependent_id)
        key = self.row_key(row)
        if dependent.lookup not in {"LIVE_AUTHORITATIVE", "TERMINAL_AUTHORITATIVE"}:
            print(
                f"[watchdog] cleanup pending for {key}: S2 {dependent_id} "
                f"lookup={dependent.lookup}; no mutation",
                flush=True,
            )
            return False
        if dependent.lookup == "TERMINAL_AUTHORITATIVE":
            return True
        if dependent.state not in WAITING_STATES:
            message = (
                f"parent S1 is unusable but S2 {dependent_id} is {dependent.state}; "
                "manifest/dependency inconsistency requires manual review"
            )
            self.state["blocked"][f"{key}|downstream"] = message
            self.log_action(row, "s2", "BLOCKED_DOWNSTREAM_STATE", old_job=dependent_id, reason=message)
            print(f"[watchdog] BLOCKED {key}: {message}", flush=True)
            return True
        if not parent_id or not dependency_has_exact_afterok(dependent.dependency, parent_id):
            message = (
                f"refusing to cancel pending S2 {dependent_id}: Dependency={dependent.dependency!r} "
                f"does not match afterok:{parent_id or '<missing>'}"
            )
            self.state["blocked"][f"{key}|downstream"] = message
            self.log_action(row, "s2", "BLOCKED_DEPENDENCY_MISMATCH", old_job=dependent_id, reason=message)
            print(f"[watchdog] BLOCKED {key}: {message}", flush=True)
            return True
        if not self.settings.auto_retry:
            print(
                f"[watchdog] cleanup needed for {key}: pending S2 {dependent_id}; {parent_reason}",
                flush=True,
            )
            return False

        result = run_command(["scancel", dependent_id])
        if result.returncode != 0:
            print(
                f"[watchdog] cleanup scancel failed for {dependent_id}: "
                f"{result.stderr.strip() or result.stdout.strip()}",
                flush=True,
            )
            return False
        deadline = time.time() + self.settings.cancel_wait_seconds
        while time.time() < deadline:
            status = self.query(row, "s2", dependent_id)
            if status.lookup == "TERMINAL_AUTHORITATIVE":
                if status.state == "CANCELLED":
                    self.log_action(
                        row,
                        "s2",
                        "CANCEL_DOWNSTREAM_PARENT_FAILED",
                        old_job=dependent_id,
                        reason=parent_reason,
                    )
                    return True
                message = f"downstream S2 became {status.state} after cleanup request"
                self.state["blocked"][f"{key}|downstream"] = message
                print(f"[watchdog] BLOCKED {key}: {message}", flush=True)
                return True
            time.sleep(2)
        print(f"[watchdog] cleanup confirmation pending for S2 {dependent_id}", flush=True)
        return False

    def cycle(self) -> tuple[bool, bool]:
        all_done = True
        any_active = False
        for row in self.managed_rows:
            s1_job = normalize_job_id(row.get("s1_job"))
            if s1_job:
                s1_info = self.query(row, "s1", s1_job)
                s1_result = self.process_job(row, "s1", s1_info)
                if s1_result == "restarted":
                    all_done = False
                    any_active = True
                    continue
                if s1_result != "completed":
                    all_done = False
                    if s1_result == "active":
                        any_active = True
                    else:
                        cleanup_done = self.quiesce_downstream(
                            row, f"S1 {s1_job} is {s1_info.state}"
                        )
                        any_active = any_active or not cleanup_done
                    continue
                s1_artifact = self.completed_artifact_status(row, "s1", s1_info)
                if s1_artifact != "completed":
                    all_done = False
                    if s1_artifact == "active":
                        any_active = True
                    else:
                        cleanup_done = self.quiesce_downstream(
                            row, f"S1 {s1_job} completed without a valid checkpoint"
                        )
                        any_active = any_active or not cleanup_done
                    continue

            s2_job = normalize_job_id(row.get("s2_job"))
            if not s2_job:
                key = self.row_key(row)
                self.state["blocked"][f"{key}|s2"] = "missing s2 job id"
                all_done = False
                continue
            s2_info = self.query(row, "s2", s2_job)
            s2_result = self.process_job(row, "s2", s2_info)
            if s2_result != "completed":
                all_done = False
                any_active = any_active or s2_result in {"active", "restarted"}
            else:
                artifact = self.completed_artifact_status(row, "s2", s2_info)
                if artifact != "completed":
                    all_done = False
                    any_active = any_active or artifact == "active"

        self.save()
        return all_done, any_active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", default=[], help="May be repeated")
    parser.add_argument(
        "--only-row",
        action="append",
        default=[],
        help="Explicit candidate:seed allowlist; may be repeated",
    )
    parser.add_argument("--resolved-manifest", default=os.environ.get("WATCH_RESOLVED_MANIFEST", ""))
    parser.add_argument("--inspect-log", default="", help="Parse one log and exit without Slurm commands")
    parser.add_argument("--once", action="store_true", default=env_bool("WATCH_ONCE", False))
    return parser.parse_args()


def collect_manifests(cli_values: list[str]) -> list[Path]:
    values = list(cli_values)
    env_many = os.environ.get("WATCH_MANIFESTS", "").strip()
    env_one = os.environ.get("WATCH_MANIFEST", "").strip()
    if env_many:
        values.extend(item for item in env_many.split(":") if item)
    if env_one:
        values.append(env_one)
    unique: list[Path] = []
    seen: set[str] = set()
    for value in values:
        resolved = str(Path(value).resolve())
        if resolved not in seen:
            unique.append(Path(resolved))
            seen.add(resolved)
    return unique


def build_settings(args: argparse.Namespace) -> Settings:
    return Settings(
        poll_seconds=env_int("WATCH_POLL_SECONDS", 300),
        startup_stale_minutes=env_int("WATCH_STARTUP_STALE_MINUTES", 120),
        data_stale_minutes=env_int("WATCH_DATA_STALE_MINUTES", 180),
        train_stale_minutes=env_int("WATCH_TRAIN_STALE_MINUTES", 180),
        validation_stale_minutes=env_int("WATCH_VALIDATION_STALE_MINUTES", 360),
        confirmations=env_int("WATCH_CONFIRMATIONS", 2),
        max_retries=env_int("WATCH_MAX_RETRIES", 2),
        auto_retry=env_bool("WATCH_AUTO_RETRY", False),
        exclude_failed_nodes=env_bool("WATCH_EXCLUDE_FAILED_NODES", True),
        once=bool(args.once),
        recheck_seconds=env_int("WATCH_RECHECK_SECONDS", 15),
        cancel_wait_seconds=env_int("WATCH_CANCEL_WAIT_SECONDS", 120),
        checkpoint_grace_minutes=env_int("WATCH_CHECKPOINT_GRACE_MINUTES", 30),
        sbatch_script=os.environ.get(
            "WATCH_SBATCH_SCRIPT",
            "/public/home/putianshu/vis_mlp/train/sub_static_rnn_lowvis_loss_matrix.slurm",
        ),
        local_cache_dir=os.environ.get("WATCH_LOCAL_CACHE_DIR", "/tmp"),
        only_rows=parse_only_rows(args.only_row),
        retry_exports=parse_retry_exports(os.environ.get("WATCH_RETRY_EXPORTS", "")),
    )


def require_slurm_commands(auto_retry: bool) -> None:
    required = ["squeue", "scontrol", "sacct"]
    if auto_retry:
        required.extend(["sbatch", "scancel"])
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"required Slurm commands are unavailable: {missing}")


def write_combined_resolved(watches: list[ManifestWatch], output: Path) -> None:
    if not watches:
        return
    fieldnames = watches[0].fieldnames
    combined: list[dict[str, str]] = []
    logical_keys: set[tuple[str, str]] = set()
    for watch in watches:
        if watch.fieldnames != fieldnames:
            raise ValueError("cannot combine resolved manifests with different columns")
        for row in watch.managed_rows:
            key = (row["candidate_id"], row["seed"])
            if key in logical_keys:
                raise ValueError(f"duplicate candidate/seed across resolved manifests: {key}")
            logical_keys.add(key)
            combined.append(dict(row))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_manifest(output, fieldnames, combined)


def main() -> int:
    args = parse_args()
    if args.inspect_log:
        progress = parse_progress(Path(args.inspect_log))
        print(json.dumps(asdict(progress), indent=2, ensure_ascii=False))
        return 0

    manifests = collect_manifests(args.manifest)
    if not manifests:
        raise SystemExit("set WATCH_MANIFEST/WATCH_MANIFESTS or pass --manifest")
    if args.resolved_manifest and len(manifests) != 1:
        raise SystemExit("WATCH_RESOLVED_MANIFEST is supported only with one manifest")

    settings = build_settings(args)
    require_slurm_commands(settings.auto_retry)
    if settings.confirmations < 1 or settings.max_retries < 0 or settings.poll_seconds < 1:
        raise ValueError("confirmations and poll_seconds must be positive; max_retries must be non-negative")
    stale_limits = (
        settings.startup_stale_minutes,
        settings.data_stale_minutes,
        settings.train_stale_minutes,
        settings.validation_stale_minutes,
        settings.checkpoint_grace_minutes,
    )
    if any(value < 1 for value in stale_limits):
        raise ValueError("stale and checkpoint-grace limits must be positive")
    if settings.auto_retry:
        if not settings.only_rows:
            raise ValueError("WATCH_AUTO_RETRY=1 requires an explicit WATCH_ONLY_ROWS allowlist")
        if settings.poll_seconds < 60 or settings.confirmations < 2 or settings.recheck_seconds < 10:
            raise ValueError(
                "auto retry requires WATCH_POLL_SECONDS>=60, WATCH_CONFIRMATIONS>=2, "
                "and WATCH_RECHECK_SECONDS>=10"
            )
        if min(stale_limits[:4]) < 30:
            raise ValueError("auto retry requires every semantic stale limit to be at least 30 minutes")
    if settings.auto_retry and not Path(settings.sbatch_script).is_file():
        raise FileNotFoundError(f"watch sbatch script not found: {settings.sbatch_script}")

    watches: list[ManifestWatch] = []
    try:
        for path in manifests:
            watches.append(
                ManifestWatch(path, settings, args.resolved_manifest if len(manifests) == 1 else "")
            )
        row_owners: dict[str, Path] = {}
        job_owners: dict[str, str] = {}
        for watch in watches:
            for row in watch.managed_rows:
                logical = f"{row['candidate_id']}:{row['seed']}"
                if logical in row_owners:
                    raise ValueError(
                        f"managed row {logical} appears in both {row_owners[logical]} and {watch.original}"
                    )
                row_owners[logical] = watch.original
                for stage in ("s1", "s2"):
                    job_id = normalize_job_id(row.get(f"{stage}_job"))
                    if not job_id:
                        continue
                    owner = f"{logical}:{stage}"
                    if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
                        raise ValueError(f"invalid Slurm JobID for {owner}: {job_id!r}")
                    if job_id in job_owners:
                        raise ValueError(
                            f"Slurm job {job_id} is shared by {job_owners[job_id]} and {owner}"
                        )
                    job_owners[job_id] = owner
        missing_rows = set(settings.only_rows) - set(row_owners)
        if missing_rows:
            raise ValueError(f"WATCH_ONLY_ROWS entries not found exactly once: {sorted(missing_rows)}")
        combined_raw = os.environ.get("WATCH_COMBINED_RESOLVED_MANIFEST", "").strip()
        combined_path = Path(combined_raw).resolve() if combined_raw else None
        if combined_path is not None:
            protected = {watch.original for watch in watches} | {watch.resolved for watch in watches}
            if combined_path in protected:
                raise ValueError(
                    "combined resolved manifest must not overwrite an input or per-manifest resolved file"
                )
            write_combined_resolved(watches, combined_path)
        print(f"[watchdog] started at {now_iso()}", flush=True)
        print(
            f"[watchdog] settings={json.dumps(asdict(settings), ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )
        for watch in watches:
            print(f"[watchdog] original={watch.original}", flush=True)
            print(f"[watchdog] resolved={watch.resolved}", flush=True)
            managed_labels = [
                f"{row['candidate_id']}:{row['seed']}" for row in watch.managed_rows
            ]
            print(
                f"[watchdog] managed_rows={managed_labels}",
                flush=True,
            )
        if combined_path is not None:
            print(f"[watchdog] combined_resolved={combined_path}", flush=True)

        while True:
            all_done = True
            any_active = False
            for watch in watches:
                done, active = watch.cycle()
                all_done = all_done and done
                any_active = any_active or active
            if combined_path is not None:
                write_combined_resolved(watches, combined_path)
            if all_done:
                print(f"[watchdog] all managed S2 jobs completed at {now_iso()}", flush=True)
                return 0
            if settings.once:
                return 0
            if not any_active:
                print("[watchdog] no active jobs remain and at least one row is blocked", flush=True)
                return 2
            time.sleep(settings.poll_seconds)
    finally:
        for watch in reversed(watches):
            watch.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[watchdog] interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)
