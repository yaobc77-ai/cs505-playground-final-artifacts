from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from env.toytextmdp import load_tasks

from .diagnose_reflection import MODE_SPECS


def _seed_roots(run_root: Path) -> list[tuple[int, Path]]:
    seed_dirs = sorted(
        (
            path
            for path in run_root.iterdir()
            if path.is_dir() and path.name.startswith("seed_")
        ),
        key=lambda path: path.name,
    )
    if not seed_dirs:
        return [(0, run_root)]
    roots: list[tuple[int, Path]] = []
    for path in seed_dirs:
        try:
            seed = int(path.name.split("_", 1)[1])
        except ValueError:
            continue
        roots.append((seed, path))
    return roots


def _load_jsonl_with_validation(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    records: list[dict[str, Any]] = []
    corrupted_lines: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                corrupted_lines.append(lineno)
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                corrupted_lines.append(lineno)
    return records, corrupted_lines


def _resolve_tasks_path(records_path: Path, records: list[dict[str, Any]]) -> Path | None:
    meta_record = next(
        (record for record in records if record.get("record_type") == "run_meta"),
        None,
    )
    raw_tasks_path = str((meta_record or {}).get("tasks_path", "")).strip()
    if not raw_tasks_path:
        return None
    tasks_path = Path(raw_tasks_path)
    if tasks_path.exists():
        return tasks_path
    if tasks_path.is_absolute():
        return None
    candidate = (records_path.parent / tasks_path).resolve()
    return candidate if candidate.exists() else None


def _task_ids_from_records(records: list[dict[str, Any]], record_type: str) -> set[str]:
    return {
        str(record.get("task_id"))
        for record in records
        if record.get("record_type") == record_type and record.get("task_id")
    }


def _compact_task_value(values: list[str]) -> str:
    return "0" if not values else ", ".join(values)


def _compact_corrupted_value(corrupted_by_file: dict[str, list[int]]) -> str:
    if not corrupted_by_file:
        return "0"
    parts = [
        f"{name}:{','.join(str(line) for line in lines)}"
        for name, lines in sorted(corrupted_by_file.items())
    ]
    return "; ".join(parts)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_run_receipt(
    *,
    run_root: Path,
    owner: str = "runner",
    lane: str = "discovery",
    shell_timeout_observed: bool = False,
) -> dict[str, Any]:
    present_modes: dict[str, dict[str, Any]] = {}
    tasks_paths: set[Path] = set()
    corrupted_by_file: dict[str, list[int]] = {}

    for seed, seed_root in _seed_roots(run_root):
        for mode_name, spec in MODE_SPECS.items():
            records_path = seed_root / spec["records"]
            if not records_path.exists():
                continue
            records, corrupted_lines = _load_jsonl_with_validation(records_path)
            if corrupted_lines:
                corrupted_by_file[f"{seed_root.name}/{records_path.name}"] = corrupted_lines
            meta_record = next(
                (record for record in records if record.get("record_type") == "run_meta"),
                None,
            )
            tasks_path = _resolve_tasks_path(records_path, records)
            if tasks_path is not None:
                tasks_paths.add(tasks_path)
            present_modes[f"{seed}:{mode_name}"] = {
                "seed": seed,
                "seed_root": seed_root,
                "mode_name": mode_name,
                "records_path": records_path,
                "records": records,
                "meta_record": meta_record,
                "episode_records": [
                    record for record in records if record.get("record_type") == "episode"
                ],
                "episode_task_ids": _task_ids_from_records(records, "episode"),
                "corrupted_lines": corrupted_lines,
            }

    if not present_modes:
        raise FileNotFoundError(f"No run records found under {run_root}")

    expected_task_ids: set[str] = set()
    tasks_path_value: Path | None = None
    expected_episode_count = 0
    if len(tasks_paths) == 1:
        tasks_path_value = next(iter(tasks_paths))
        expected_task_ids = {
            str(task.get("task_id"))
            for task in load_tasks(tasks_path_value)
            if task.get("task_id")
        }
        expected_episode_count = len(expected_task_ids)

    missing_tasks: set[str] = set()
    extra_tasks: set[str] = set()
    pairing_failures: list[str] = []
    cross_mode_mismatches: list[str] = []
    baseline_key: str | None = None
    baseline_task_ids: set[str] | None = None

    for key, mode_info in sorted(present_modes.items()):
        episode_task_ids = set(mode_info["episode_task_ids"])
        if baseline_task_ids is None:
            baseline_task_ids = episode_task_ids
            baseline_key = key
        elif episode_task_ids != baseline_task_ids:
            cross_mode_mismatches.append(f"{key}!={baseline_key}")
        if expected_task_ids:
            missing_tasks.update(expected_task_ids - episode_task_ids)
            extra_tasks.update(episode_task_ids - expected_task_ids)
        if mode_info["meta_record"] is None:
            pairing_failures.append(f"{key}:missing_run_meta")
        if not episode_task_ids:
            pairing_failures.append(f"{key}:missing_episode_records")

    if len(tasks_paths) > 1:
        pairing_failures.append("multiple_tasks_paths")

    valid_episode_counts = {
        key: len(info["episode_records"])
        for key, info in sorted(present_modes.items())
    }
    artifact_hashes: dict[str, str] = {}
    if tasks_path_value is not None and tasks_path_value.exists():
        artifact_hashes["tasks_path"] = _sha256(tasks_path_value)
    for key, info in sorted(present_modes.items()):
        artifact_hashes[f"{key}:{info['records_path'].name}"] = _sha256(info["records_path"])

    jsonl_integrity = not corrupted_by_file
    task_id_pairing = not (missing_tasks or extra_tasks or pairing_failures or cross_mode_mismatches)
    hash_complete = bool(artifact_hashes) and (
        "tasks_path" in artifact_hashes if tasks_path_value is not None else True
    )
    count_alignment = (
        expected_episode_count == 0
        or all(count == expected_episode_count for count in valid_episode_counts.values())
    )
    shell_timeout_accepted_by_integrity_rule = bool(
        shell_timeout_observed and jsonl_integrity and task_id_pairing and hash_complete and count_alignment
    )
    reports_generated = [
        name
        for name in (
            "diagnosis.json",
            "diagnosis_report.md",
            "diagnosis_step_rows.jsonl",
            "verify_survival_rows.jsonl",
            "pruning_path_audit.md",
            "pre_selector_pruning_rows.jsonl",
            "pre_selector_pruning_audit.md",
            "blocked_step_audit_rows.jsonl",
            "blocked_step_audit.md",
        )
        if (run_root / name).exists()
    ]
    meta_timestamps = [
        str(info["meta_record"].get("timestamp"))
        for info in present_modes.values()
        if info["meta_record"] and info["meta_record"].get("timestamp")
    ]

    return {
        "run_name": run_root.name,
        "date": min(meta_timestamps) if meta_timestamps else "",
        "owner": owner,
        "lane": lane,
        "run_completed": all(
            info["meta_record"] is not None and not info["corrupted_lines"] and info["episode_task_ids"]
            for info in present_modes.values()
        ),
        "jsonl_integrity": "pass" if jsonl_integrity else "fail",
        "task_id_pairing": "pass" if task_id_pairing else "fail",
        "expected_episode_count": expected_episode_count,
        "valid_episode_counts": valid_episode_counts,
        "missing_tasks": sorted(missing_tasks),
        "extra_tasks": sorted(extra_tasks),
        "corrupted_lines": corrupted_by_file,
        "artifact_hashes": artifact_hashes,
        "hash_complete": bool(hash_complete),
        "shell_timeout_observed": bool(shell_timeout_observed),
        "shell_timeout_accepted_by_integrity_rule": shell_timeout_accepted_by_integrity_rule,
        "reports_generated": reports_generated,
        "safe_for_interpretation": bool(jsonl_integrity and task_id_pairing),
        "tasks_path": str(tasks_path_value) if tasks_path_value else None,
        "mode_keys": sorted(present_modes),
        "summary_notes": "; ".join(
            note
            for note in [
                f"tasks_path={tasks_path_value}" if tasks_path_value else "tasks_path=unresolved",
                (
                    f"expected_episode_count={expected_episode_count},"
                    f"valid_episode_counts={','.join(f'{key}:{count}' for key, count in valid_episode_counts.items())}"
                    if valid_episode_counts
                    else ""
                ),
                (
                    "shell_timeout_accepted_by_integrity_rule"
                    if shell_timeout_accepted_by_integrity_rule
                    else ""
                ),
                f"cross_mode_mismatches={','.join(cross_mode_mismatches)}" if cross_mode_mismatches else "",
                f"pairing_failures={','.join(pairing_failures)}" if pairing_failures else "",
                f"extra_tasks={','.join(sorted(extra_tasks))}" if extra_tasks else "",
            ]
            if note
        ) or "All present mode files validated and pair cleanly against the task file.",
    }


def render_receipt_markdown(receipt: dict[str, Any]) -> str:
    reports = receipt.get("reports_generated") or []
    reports_value = ", ".join(str(item) for item in reports) if reports else "0"
    lines = [
        "# Post-Run Receipt",
        "",
        f"- run_name: {receipt['run_name']}",
        f"- date: {receipt.get('date', '')}",
        f"- owner: {receipt.get('owner', 'runner')}",
        f"- run_completed: {_bool_text(bool(receipt.get('run_completed')))}",
        f"- jsonl_integrity: {receipt['jsonl_integrity']}",
        f"- task_id_pairing: {receipt['task_id_pairing']}",
        f"- expected_episode_count: {receipt.get('expected_episode_count', 0)}",
        f"- valid_episode_counts: {_compact_task_value([f'{key}:{count}' for key, count in dict(receipt.get('valid_episode_counts') or {}).items()])}",
        f"- missing_tasks: {_compact_task_value(list(receipt.get('missing_tasks') or []))}",
        f"- corrupted_lines: {_compact_corrupted_value(dict(receipt.get('corrupted_lines') or {}))}",
        f"- hash_complete: {_bool_text(bool(receipt.get('hash_complete')))}",
        f"- artifact_hashes: {_compact_task_value([f'{key}:{value}' for key, value in dict(receipt.get('artifact_hashes') or {}).items()])}",
        f"- shell_timeout_observed: {_bool_text(bool(receipt.get('shell_timeout_observed')))}",
        f"- shell_timeout_accepted_by_integrity_rule: {_bool_text(bool(receipt.get('shell_timeout_accepted_by_integrity_rule')))}",
        f"- reports_generated: {reports_value}",
        f"- safe_for_interpretation: {_bool_text(bool(receipt.get('safe_for_interpretation')))}",
        f"- lane: {receipt.get('lane', 'discovery')}",
        f"- summary_notes: {receipt.get('summary_notes', '')}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a run root and emit a post-run receipt.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--owner", type=str, default="runner")
    parser.add_argument("--lane", type=str, default="discovery")
    parser.add_argument("--shell-timeout-observed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = collect_run_receipt(
        run_root=args.run_root,
        owner=args.owner,
        lane=args.lane,
        shell_timeout_observed=args.shell_timeout_observed,
    )
    markdown = render_receipt_markdown(receipt)
    if args.out is not None:
        args.out.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
