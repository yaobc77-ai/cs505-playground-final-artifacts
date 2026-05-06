from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from data.generate_tasks import generate_tasks_file
from eval.diagnose_reflection import run_diagnosis
from eval.run_suite import run_suite
from eval.stats import bootstrap_interval

DEFAULT_SWEEP_CONFIGS = "0.085:0.375,0.090:0.400,0.095:0.425"
SUMMARY_FIELDS = ("success_rate", "avg_reward", "avg_steps")
PROCESS_FIELDS = ("avg_asks_per_episode", "choose_without_asking_rate", "reflection_admission_rate")
DIAGNOSIS_FIELDS = (
    "selected_ask_rate",
    "selected_choose_rate",
    "selected_choose_without_prior_ask_rate",
    "regretful_ask_rate",
    "choose_regret_rate",
    "over_ask_cost_mean_per_step",
    "under_ask_cost_mean_per_step",
)
DELTA_FIELDS = (
    "avg_reward",
    "success_rate",
    "avg_steps",
    "avg_asks_per_episode",
    "regretful_ask_rate",
    "choose_regret_rate",
    "under_ask_cost_mean_per_step",
)
BASELINE_FILES = ("full.jsonl", "full_memory.jsonl", "full_audit.jsonl")


def _config_to_label(penalty_base_raw: str, penalty_memory_scale_raw: str) -> str:
    return (
        f"base_{penalty_base_raw}_scale_{penalty_memory_scale_raw}"
        .replace(".", "p")
        .replace("-", "m")
    )


def _selector_display(
    selector_config: dict[str, Any],
    selector_display: dict[str, Any] | None = None,
) -> dict[str, str]:
    selector_display = dict(selector_display or {})
    return {
        "penalty_base": str(
            selector_display.get("penalty_base", f"{float(selector_config['penalty_base']):.3f}")
        ),
        "penalty_memory_scale": str(
            selector_display.get(
                "penalty_memory_scale",
                f"{float(selector_config['penalty_memory_scale']):.3f}",
            )
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _episode_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("record_type") == "episode"]


def _step_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("record_type") == "step"]


def _summary(records: list[dict[str, Any]]) -> dict[str, float]:
    episodes = _episode_records(records)
    if not episodes:
        return {"success_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0}
    return {
        "success_rate": round(
            sum(float(bool(record.get("success"))) for record in episodes) / len(episodes),
            4,
        ),
        "avg_reward": round(
            sum(float(record.get("total_reward", 0.0)) for record in episodes) / len(episodes),
            4,
        ),
        "avg_steps": round(
            sum(float(record.get("steps", 0.0)) for record in episodes) / len(episodes),
            4,
        ),
    }


def _process_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    episodes = _episode_records(records)
    steps_by_task: dict[str, list[dict[str, Any]]] = {}
    for step in _step_records(records):
        steps_by_task.setdefault(str(step.get("task_id")), []).append(step)

    if not episodes:
        return {
            "avg_asks_per_episode": 0.0,
            "choose_without_asking_rate": 0.0,
            "reflection_admission_rate": 0.0,
        }

    ask_counts: list[float] = []
    choose_without_asking: list[float] = []
    reflection_admitted: list[float] = []
    for episode in episodes:
        task_id = str(episode.get("task_id"))
        task_steps = steps_by_task.get(task_id, [])
        asks = sum(1 for step in task_steps if str(step.get("action", "")).startswith("ASK"))
        ask_counts.append(float(asks))
        choose_without_asking.append(float(asks == 0))
        reflection_admitted.append(float(bool(episode.get("reflection_admitted"))))

    episode_count = len(episodes)
    return {
        "avg_asks_per_episode": round(sum(ask_counts) / episode_count, 4),
        "choose_without_asking_rate": round(sum(choose_without_asking) / episode_count, 4),
        "reflection_admission_rate": round(sum(reflection_admitted) / episode_count, 4),
    }


def _parse_configs(raw: str) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            base_raw, scale_raw = part.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid selector penalty config '{part}'. Expected base:memory_scale."
            ) from exc
        base_text = base_raw.strip()
        scale_text = scale_raw.strip()
        penalty_base = float(base_text)
        penalty_memory_scale = float(scale_text)
        label = _config_to_label(base_text, scale_text)
        configs.append(
            {
                "label": label,
                "selector_config": {
                    "penalty_base": penalty_base,
                    "penalty_memory_scale": penalty_memory_scale,
                },
                "selector_display": {
                    "penalty_base": base_text,
                    "penalty_memory_scale": scale_text,
                },
            }
        )
    if not configs:
        raise ValueError("At least one selector penalty config is required.")
    return configs


def _config_from_result(label: str, result: dict[str, Any]) -> dict[str, Any]:
    config_block = result.get("configs", {}).get(label)
    if config_block is not None:
        selector_config = config_block.get("selector_config", {})
        return {
            "label": label,
            "selector_config": {
                "penalty_base": float(selector_config["penalty_base"]),
                "penalty_memory_scale": float(selector_config["penalty_memory_scale"]),
            },
            "selector_display": _selector_display(
                selector_config,
                config_block.get("selector_display"),
            ),
        }

    configs_block = result.get("config", {}).get("configs", [])
    if len(configs_block) == 1:
        config = configs_block[0]
        selector_config = config.get("selector_config", {})
        return {
            "label": str(config.get("label", label)),
            "selector_config": {
                "penalty_base": float(selector_config["penalty_base"]),
                "penalty_memory_scale": float(selector_config["penalty_memory_scale"]),
            },
            "selector_display": _selector_display(
                selector_config,
                config.get("selector_display"),
            ),
        }

    raise ValueError(f"Could not determine selector config for {label}.")


def _discover_configs_from_aggregate_root(aggregate_root: Path) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for child in sorted(path for path in aggregate_root.iterdir() if path.is_dir()):
        if child.name == "baseline":
            continue
        final_path = child / "sweep_results.json"
        progress_path = child / "sweep_progress.json"
        if not final_path.exists() and not progress_path.exists():
            continue
        result = _load_existing_sweep_result(child)
        configs.append(_config_from_result(child.name, result))

    if not configs:
        raise FileNotFoundError(f"No selector-penalty config results found under {aggregate_root}")
    return configs


def _aggregate_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    return bootstrap_interval(values)


def _aggregate_field_block(rows: list[dict[str, Any]], field_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        field_name: _aggregate_values([float(row[field_name]) for row in rows])
        for field_name in field_names
    }


def _delta_block(
    selector_block: dict[str, float],
    full_block: dict[str, float],
    field_names: tuple[str, ...],
) -> dict[str, float]:
    return {
        field_name: round(float(selector_block[field_name]) - float(full_block[field_name]), 4)
        for field_name in field_names
    }


def _format_interval(interval: dict[str, float], digits: int = 3) -> str:
    return f"{interval['mean']:.{digits}f} [{interval['ci_low']:.{digits}f}, {interval['ci_high']:.{digits}f}]"


def _pick_balanced_config(results: dict[str, Any]) -> str | None:
    candidates: list[tuple[float, float, str]] = []
    for label, block in results.items():
        reward_diff = block["deltas_vs_full"]["avg_reward"]["mean"]
        success_diff = block["deltas_vs_full"]["success_rate"]["mean"]
        choose_regret = block["diagnosis"]["choose_regret_rate"]["mean"]
        regretful_ask = block["diagnosis"]["regretful_ask_rate"]["mean"]
        if reward_diff >= -0.02 and success_diff >= -0.02:
            candidates.append((regretful_ask, choose_regret, label))
    if candidates:
        candidates.sort()
        return candidates[0][2]

    if not results:
        return None
    return max(
        results.items(),
        key=lambda item: (
            item[1]["deltas_vs_full"]["avg_reward"]["mean"],
            -item[1]["diagnosis"]["regretful_ask_rate"]["mean"],
        ),
    )[0]


def _render_sweep_report(
    *,
    out_root: Path,
    task_count: int | None,
    seed: int | None,
    repeats: int,
    completed_repeats: int,
    full_baseline: dict[str, Any],
    config_results: dict[str, Any],
    recommended_config: str | None,
    baseline_source_label: str | None = None,
) -> str:
    lines = [
        "# Selector Penalty Calibration Sweep",
        "",
        f"Run root: `{out_root}`",
        f"Task count: {task_count if task_count is not None else 'unknown'}",
        f"Task seed: {seed if seed is not None else 'unknown'}",
        f"Completed repeats: {completed_repeats}/{repeats}",
    ]
    if baseline_source_label:
        lines.append(f"Full baseline source: `{baseline_source_label}`")
    lines.extend(
        [
            "",
            "## Full Baseline",
            "",
            f"- Reward: {_format_interval(full_baseline['summary']['avg_reward'])}",
            f"- Success: {_format_interval(full_baseline['summary']['success_rate'])}",
            f"- Steps: {_format_interval(full_baseline['summary']['avg_steps'])}",
            f"- Avg asks: {_format_interval(full_baseline['process']['avg_asks_per_episode'])}",
            f"- Regretful ASK rate: {_format_interval(full_baseline['diagnosis']['regretful_ask_rate'])}",
            f"- Choose regret rate: {_format_interval(full_baseline['diagnosis']['choose_regret_rate'])}",
            f"- Over-ask cost/step: {_format_interval(full_baseline['diagnosis']['over_ask_cost_mean_per_step'])}",
            f"- Under-ask cost/step: {_format_interval(full_baseline['diagnosis']['under_ask_cost_mean_per_step'])}",
            "",
            "## Config Comparison",
            "",
            "| Config | Completed repeats | Reward | Success | Steps | Avg asks | Regretful ASK | Choose regret | Over-ask cost | Under-ask cost | Reward diff vs full | Ask diff vs full |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for label, block in config_results.items():
        selector_config = block["selector_config"]
        selector_display = _selector_display(
            selector_config,
            block.get("selector_display"),
        )
        display_label = (
            f"base={selector_display['penalty_base']}, scale={selector_display['penalty_memory_scale']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    display_label,
                    str(int(block.get("completed_repeats", 0))),
                    _format_interval(block["summary"]["avg_reward"]),
                    _format_interval(block["summary"]["success_rate"]),
                    _format_interval(block["summary"]["avg_steps"]),
                    _format_interval(block["process"]["avg_asks_per_episode"]),
                    _format_interval(block["diagnosis"]["regretful_ask_rate"]),
                    _format_interval(block["diagnosis"]["choose_regret_rate"]),
                    _format_interval(block["diagnosis"]["over_ask_cost_mean_per_step"]),
                    _format_interval(block["diagnosis"]["under_ask_cost_mean_per_step"]),
                    _format_interval(block["deltas_vs_full"]["avg_reward"]),
                    _format_interval(block["deltas_vs_full"]["avg_asks_per_episode"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    if recommended_config is None:
        lines.append("- No selector-penalty configs were available.")
    else:
        config = config_results[recommended_config]
        selector_display = _selector_display(
            config["selector_config"],
            config.get("selector_display"),
        )
        lines.append(
            "- Balanced candidate: "
            + f"`base={selector_display['penalty_base']}, scale={selector_display['penalty_memory_scale']}`"
        )
        lines.append(
            "- This pick prioritizes lower regretful ASK while avoiding a large mean reward or success drop."
        )

    lines.append("")
    return "\n".join(lines)


def _build_repeat_row(
    *,
    repeat_index: int,
    summary: dict[str, float],
    process: dict[str, float],
    diagnosis_block: dict[str, Any],
) -> dict[str, Any]:
    return {
        "repeat_index": repeat_index,
        "summary": summary,
        "process": process,
        "diagnosis": {
            field_name: float(diagnosis_block.get(field_name, 0.0))
            for field_name in DIAGNOSIS_FIELDS
        },
    }


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _baseline_repeat_root(baseline_root: Path, repeat_index: int) -> Path:
    return baseline_root / f"repeat_{repeat_index:02d}"


def _load_existing_sweep_result(result_root: Path) -> dict[str, Any]:
    final_path = result_root / "sweep_results.json"
    progress_path = result_root / "sweep_progress.json"
    if final_path.exists():
        return _load_json(final_path)
    if progress_path.exists():
        return _load_json(progress_path)
    raise FileNotFoundError(f"Missing sweep result under {result_root}")


def _build_sweep_results(
    *,
    out_root: Path,
    task_count: int | None,
    seed: int | None,
    repeats: int,
    configs: list[dict[str, Any]],
    full_runs: list[dict[str, Any]],
    config_runs: dict[str, list[dict[str, Any]]],
    baseline_source_label: str | None = None,
) -> dict[str, Any]:
    full_baseline = {
        "summary": _aggregate_field_block(
            [row["summary"] for row in full_runs],
            SUMMARY_FIELDS,
        ),
        "process": _aggregate_field_block(
            [row["process"] for row in full_runs],
            PROCESS_FIELDS,
        ),
        "diagnosis": _aggregate_field_block(
            [row["diagnosis"] for row in full_runs],
            DIAGNOSIS_FIELDS,
        ),
        "per_repeat": full_runs,
        "completed_repeats": len(full_runs),
    }

    config_results: dict[str, Any] = {}
    for config in configs:
        rows = config_runs.get(config["label"], [])
        if not rows:
            continue
        config_results[config["label"]] = {
            "selector_config": dict(config["selector_config"]),
            "selector_display": _selector_display(
                config["selector_config"],
                config.get("selector_display"),
            ),
            "summary": _aggregate_field_block(
                [row["summary"] for row in rows],
                SUMMARY_FIELDS,
            ),
            "process": _aggregate_field_block(
                [row["process"] for row in rows],
                PROCESS_FIELDS,
            ),
            "diagnosis": _aggregate_field_block(
                [row["diagnosis"] for row in rows],
                DIAGNOSIS_FIELDS,
            ),
            "deltas_vs_full": _aggregate_field_block(
                [row["deltas_vs_full"] for row in rows],
                DELTA_FIELDS,
            ),
            "per_repeat": rows,
            "completed_repeats": len(rows),
        }

    completed_candidates = [len(full_runs)]
    completed_candidates.extend(len(rows) for rows in config_runs.values())
    completed_repeats = min(completed_candidates) if completed_candidates else 0
    recommended_config = _pick_balanced_config(config_results)

    return {
        "config": {
            "out_root": str(out_root),
            "task_count": task_count,
            "seed": seed,
            "repeats": repeats,
            "completed_repeats": completed_repeats,
            "complete": completed_repeats >= repeats,
            "configs": configs,
            "baseline_source_label": baseline_source_label,
        },
        "full_baseline": full_baseline,
        "configs": config_results,
        "recommended_config": recommended_config,
    }


def _write_sweep_outputs(
    *,
    out_root: Path,
    sweep_results: dict[str, Any],
    final: bool,
) -> None:
    report = _render_sweep_report(
        out_root=out_root,
        task_count=sweep_results["config"].get("task_count"),
        seed=sweep_results["config"].get("seed"),
        repeats=int(sweep_results["config"]["repeats"]),
        completed_repeats=int(sweep_results["config"]["completed_repeats"]),
        full_baseline=sweep_results["full_baseline"],
        config_results=sweep_results["configs"],
        recommended_config=sweep_results.get("recommended_config"),
        baseline_source_label=sweep_results["config"].get("baseline_source_label"),
    )

    (out_root / "sweep_progress.json").write_text(
        json.dumps(sweep_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_root / "sweep_progress.md").write_text(report, encoding="utf-8")

    if final:
        (out_root / "sweep_results.json").write_text(
            json.dumps(sweep_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_root / "sweep_report.md").write_text(report, encoding="utf-8")


def _baseline_diagnosis_row(repeat_root: Path, repeat_index: int) -> dict[str, Any]:
    diagnosis, _ = run_diagnosis(run_root=repeat_root)
    full_block = diagnosis["modes"].get("full", {}).get("overall", {})
    full_records = _load_jsonl(repeat_root / "full.jsonl")
    return _build_repeat_row(
        repeat_index=repeat_index,
        summary=_summary(full_records),
        process=_process_summary(full_records),
        diagnosis_block=full_block,
    )


def _run_baseline_only(
    *,
    out_root: Path,
    tasks_path: Path | None,
    task_count: int,
    seed: int,
    repeats: int,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_tasks_path = tasks_path or (out_root / "tasks.jsonl")
    if tasks_path is None:
        generate_tasks_file(resolved_tasks_path, count=task_count, seed=seed)
    elif resolved_tasks_path != out_root / "tasks.jsonl":
        _copy_if_exists(resolved_tasks_path, out_root / "tasks.jsonl")

    full_runs: list[dict[str, Any]] = []
    config_runs: dict[str, list[dict[str, Any]]] = {}

    for repeat_index in range(repeats):
        repeat_root = out_root / f"repeat_{repeat_index:02d}"
        repeat_root.mkdir(parents=True, exist_ok=True)
        full_path = repeat_root / "full.jsonl"
        run_suite(
            tasks_path=resolved_tasks_path,
            mode="full",
            out_path=full_path,
            seed=seed,
            repeat_index=repeat_index,
        )
        full_runs.append(_baseline_diagnosis_row(repeat_root, repeat_index))

        partial_results = _build_sweep_results(
            out_root=out_root,
            task_count=task_count,
            seed=seed,
            repeats=repeats,
            configs=[],
            full_runs=full_runs,
            config_runs=config_runs,
            baseline_source_label="baseline",
        )
        _write_sweep_outputs(out_root=out_root, sweep_results=partial_results, final=False)

    final_results = _build_sweep_results(
        out_root=out_root,
        task_count=task_count,
        seed=seed,
        repeats=repeats,
        configs=[],
        full_runs=full_runs,
        config_runs=config_runs,
        baseline_source_label="baseline",
    )
    _write_sweep_outputs(out_root=out_root, sweep_results=final_results, final=True)


def _run_configs_against_shared_baseline(
    *,
    out_root: Path,
    shared_baseline_root: Path,
    configs: list[dict[str, Any]],
    repeats: int,
) -> None:
    if len(configs) != 1:
        raise ValueError(
            "Shared-baseline config runs expect exactly one config per output directory."
        )
    if not shared_baseline_root.exists():
        raise FileNotFoundError(f"Missing shared baseline root: {shared_baseline_root}")

    baseline_result = _load_existing_sweep_result(shared_baseline_root)
    baseline_runs = list(baseline_result.get("full_baseline", {}).get("per_repeat", []))
    baseline_repeat_by_index = {int(row["repeat_index"]): row for row in baseline_runs}
    task_count = baseline_result.get("config", {}).get("task_count")
    seed = baseline_result.get("config", {}).get("seed")

    out_root.mkdir(parents=True, exist_ok=True)
    baseline_tasks_path = shared_baseline_root / "tasks.jsonl"
    if baseline_tasks_path.exists():
        _copy_if_exists(baseline_tasks_path, out_root / "tasks.jsonl")

    full_runs: list[dict[str, Any]] = []
    config_runs: dict[str, list[dict[str, Any]]] = {config["label"]: [] for config in configs}

    for repeat_index in range(repeats):
        if repeat_index not in baseline_repeat_by_index:
            raise FileNotFoundError(
                f"Shared baseline is missing repeat {repeat_index:02d} in {shared_baseline_root}"
            )
        baseline_repeat_root = _baseline_repeat_root(shared_baseline_root, repeat_index)
        repeat_root = out_root / f"repeat_{repeat_index:02d}"
        repeat_root.mkdir(parents=True, exist_ok=True)
        for filename in BASELINE_FILES:
            _copy_if_exists(baseline_repeat_root / filename, repeat_root / filename)

        baseline_row = dict(baseline_repeat_by_index[repeat_index])
        full_runs.append(baseline_row)

        config = configs[0]
        selector_path = repeat_root / "selector_penalty.jsonl"
        run_suite(
            tasks_path=baseline_tasks_path if baseline_tasks_path.exists() else out_root / "tasks.jsonl",
            mode="selector_penalty",
            out_path=selector_path,
            seed=int(seed) if seed is not None else 7,
            selector_config=config["selector_config"],
            repeat_index=repeat_index,
        )
        diagnosis, diagnosis_report = run_diagnosis(run_root=repeat_root)
        (repeat_root / "diagnosis.json").write_text(
            json.dumps(diagnosis, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (repeat_root / "diagnosis_report.md").write_text(
            diagnosis_report,
            encoding="utf-8",
        )

        selector_records = _load_jsonl(selector_path)
        selector_summary = _summary(selector_records)
        selector_process = _process_summary(selector_records)
        selector_diagnosis = diagnosis["modes"].get("selector_penalty", {}).get("overall", {})
        selector_repeat_row = _build_repeat_row(
            repeat_index=repeat_index,
            summary=selector_summary,
            process=selector_process,
            diagnosis_block=selector_diagnosis,
        )
        selector_repeat_row["selector_config"] = dict(config["selector_config"])
        selector_repeat_row["deltas_vs_full"] = {
            **_delta_block(selector_summary, baseline_row["summary"], SUMMARY_FIELDS),
            **_delta_block(
                selector_process,
                baseline_row["process"],
                ("avg_asks_per_episode",),
            ),
            **_delta_block(
                {
                    field_name: float(selector_diagnosis.get(field_name, 0.0))
                    for field_name in (
                        "regretful_ask_rate",
                        "choose_regret_rate",
                        "under_ask_cost_mean_per_step",
                    )
                },
                baseline_row["diagnosis"],
                ("regretful_ask_rate", "choose_regret_rate", "under_ask_cost_mean_per_step"),
            ),
        }
        config_runs[config["label"]].append(selector_repeat_row)

        partial_results = _build_sweep_results(
            out_root=out_root,
            task_count=task_count,
            seed=seed,
            repeats=repeats,
            configs=configs,
            full_runs=full_runs,
            config_runs=config_runs,
            baseline_source_label=str(shared_baseline_root),
        )
        _write_sweep_outputs(out_root=out_root, sweep_results=partial_results, final=False)

    final_results = _build_sweep_results(
        out_root=out_root,
        task_count=task_count,
        seed=seed,
        repeats=repeats,
        configs=configs,
        full_runs=full_runs,
        config_runs=config_runs,
        baseline_source_label=str(shared_baseline_root),
    )
    _write_sweep_outputs(out_root=out_root, sweep_results=final_results, final=True)


def _aggregate_existing_sweep_results(
    *,
    aggregate_root: Path,
    shared_baseline_root: Path,
    configs: list[dict[str, Any]] | None,
) -> None:
    baseline_result = _load_existing_sweep_result(shared_baseline_root)
    full_runs = list(baseline_result.get("full_baseline", {}).get("per_repeat", []))
    task_count = baseline_result.get("config", {}).get("task_count")
    seed = baseline_result.get("config", {}).get("seed")
    repeats = int(baseline_result.get("config", {}).get("repeats", 0))

    if configs is None:
        configs = _discover_configs_from_aggregate_root(aggregate_root)

    config_runs: dict[str, list[dict[str, Any]]] = {}
    for config in configs:
        config_root = aggregate_root / config["label"]
        config_result = _load_existing_sweep_result(config_root)
        config_block = config_result.get("configs", {}).get(config["label"])
        if not config_block:
            other_configs = config_result.get("configs", {})
            if len(other_configs) != 1:
                raise ValueError(f"Ambiguous config block for {config['label']} in {config_root}")
            config_block = next(iter(other_configs.values()))
        config_runs[config["label"]] = list(config_block.get("per_repeat", []))

    combined_results = _build_sweep_results(
        out_root=aggregate_root,
        task_count=task_count,
        seed=seed,
        repeats=repeats,
        configs=configs,
        full_runs=full_runs,
        config_runs=config_runs,
        baseline_source_label="baseline",
    )
    _write_sweep_outputs(out_root=aggregate_root, sweep_results=combined_results, final=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or aggregate selector-penalty calibration sweeps.")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--aggregate-root", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--shared-baseline-root", type=Path, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--task-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs", type=str, default=None)
    args = parser.parse_args()

    if (args.out_root is None) == (args.aggregate_root is None):
        parser.error("Specify exactly one of --out-root or --aggregate-root.")
    if args.aggregate_root is not None and (args.baseline_only or args.shared_baseline_root is not None):
        parser.error("--aggregate-root cannot be combined with --baseline-only or --shared-baseline-root.")
    if args.baseline_only and args.shared_baseline_root is not None:
        parser.error("--baseline-only cannot be combined with --shared-baseline-root.")
    return args


def main() -> None:
    args = parse_args()
    configs = _parse_configs(args.configs) if args.configs else None

    if args.aggregate_root is not None:
        _aggregate_existing_sweep_results(
            aggregate_root=args.aggregate_root,
            shared_baseline_root=args.aggregate_root / "baseline",
            configs=configs,
        )
        return

    if args.baseline_only:
        _run_baseline_only(
            out_root=args.out_root,
            tasks_path=args.tasks,
            task_count=args.task_count,
            seed=args.seed,
            repeats=args.repeats,
        )
        return

    if configs is None:
        configs = _parse_configs(DEFAULT_SWEEP_CONFIGS)

    if args.shared_baseline_root is not None:
        if len(configs) == 1:
            _run_configs_against_shared_baseline(
                out_root=args.out_root,
                shared_baseline_root=args.shared_baseline_root,
                configs=configs,
                repeats=args.repeats,
            )
            return

        for config in configs:
            _run_configs_against_shared_baseline(
                out_root=args.out_root / config["label"],
                shared_baseline_root=args.shared_baseline_root,
                configs=[config],
                repeats=args.repeats,
            )
        _aggregate_existing_sweep_results(
            aggregate_root=args.out_root,
            shared_baseline_root=args.shared_baseline_root,
            configs=configs,
        )
        return

    _run_baseline_only(
        out_root=args.out_root / "baseline",
        tasks_path=args.tasks,
        task_count=args.task_count,
        seed=args.seed,
        repeats=args.repeats,
    )
    for config in configs:
        _run_configs_against_shared_baseline(
            out_root=args.out_root / config["label"],
            shared_baseline_root=args.out_root / "baseline",
            configs=[config],
            repeats=args.repeats,
        )
    _aggregate_existing_sweep_results(
        aggregate_root=args.out_root,
        shared_baseline_root=args.out_root / "baseline",
        configs=configs,
    )


if __name__ == "__main__":
    main()
