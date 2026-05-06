from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MAIN_MODES = ("full", "ctrl", "perm")
SUPPLEMENTARY_MODES = ("no_self_model", "no_cf_eval")
EXPERIMENTAL_MODES = (
    "selector_penalty",
    "voi_coverage_only",
    "voi_gate_only",
    "voi_memory_v1",
)
SUBJECT_MODES = (
    "subject_v0",
    "subject_no_other",
    "subject_no_commitment",
    "subject_no_memory_recompose",
)
ALL_MODES = MAIN_MODES + SUPPLEMENTARY_MODES + EXPERIMENTAL_MODES + SUBJECT_MODES


def _load_records(path: Path) -> list[dict[str, Any]]:
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


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _align_actions(records: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    return {
        (record["task_id"], int(record["step"])): record["action"]
        for record in _step_records(records)
    }


def decision_change_rates(
    full_records: list[dict[str, Any]],
    ctrl_records: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    full_actions = _align_actions(full_records)
    ctrl_actions = _align_actions(ctrl_records)
    common_keys = sorted(set(full_actions).intersection(ctrl_actions))
    if not common_keys:
        return 0.0, {}

    per_task: dict[str, list[int]] = {}
    for task_id, step in common_keys:
        changed = int(full_actions[(task_id, step)] != ctrl_actions[(task_id, step)])
        per_task.setdefault(task_id, []).append(changed)

    per_task_rate = {
        task_id: sum(changes) / len(changes)
        for task_id, changes in per_task.items()
    }
    overall = sum(sum(changes) for changes in per_task.values()) / len(common_keys)
    return overall, per_task_rate


def _match_action(action: str, avoid_action: str) -> bool:
    if avoid_action in {"ASK", "CHOOSE"}:
        return action.startswith(avoid_action)
    return action == avoid_action


def self_reinforcement_index(records: list[dict[str, Any]], window: int = 5) -> tuple[float, list[float]]:
    steps = _step_records(records)
    episodes = _episode_records(records)
    contributions: list[float] = []

    for episode in episodes:
        reflection = episode.get("reflection_written") or episode.get("reflection")
        avoid_actions = list(reflection.get("avoid_actions", [])) if reflection else []
        if not avoid_actions:
            continue

        task_id = episode["task_id"]
        episode_index = next(
            (
                index
                for index, step in enumerate(steps)
                if step["task_id"] == task_id and int(step["step"]) == int(episode["steps"])
            ),
            None,
        )
        if episode_index is None:
            continue

        before = steps[max(0, episode_index - window):episode_index]
        after = steps[episode_index:episode_index + window]
        if not after:
            continue

        for avoid_action in avoid_actions:
            before_freq = (
                sum(_match_action(step["action"], avoid_action) for step in before) / max(len(before), 1)
            )
            after_freq = (
                sum(_match_action(step["action"], avoid_action) for step in after) / len(after)
            )
            contributions.append(before_freq - after_freq)

    return (sum(contributions) / len(contributions), contributions) if contributions else (0.0, [])


def _episode_series(records: list[dict[str, Any]], field: str) -> list[float]:
    return [float(record[field]) for record in _episode_records(records)]


def _summary(records: list[dict[str, Any]]) -> dict[str, float]:
    episodes = _episode_records(records)
    if not episodes:
        return {"success_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0}
    success_rate = sum(int(record["success"]) for record in episodes) / len(episodes)
    avg_reward = sum(float(record["total_reward"]) for record in episodes) / len(episodes)
    avg_steps = sum(float(record["steps"]) for record in episodes) / len(episodes)
    return {
        "success_rate": round(success_rate, 4),
        "avg_reward": round(avg_reward, 4),
        "avg_steps": round(avg_steps, 4),
    }


def _task_tags_to_metadata(task_tags: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for tag in task_tags:
        if ":" not in str(tag):
            continue
        key, raw_value = str(tag).split(":", 1)
        if key == "difficulty":
            metadata["difficulty"] = raw_value
        elif key == "noise":
            try:
                metadata["noise_level"] = float(raw_value)
            except ValueError:
                metadata["noise_level"] = raw_value
        elif key == "benchmark":
            metadata["benchmark_family"] = raw_value
        elif key == "arc":
            metadata["arc_id"] = raw_value
        elif key == "episode":
            try:
                metadata["episode_index"] = int(raw_value)
            except ValueError:
                metadata["episode_index"] = raw_value
    return metadata


def _task_metadata(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata_by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        if not task_id or task_id in metadata_by_task:
            continue
        task_tags = list(record.get("task_tags", []))
        if not task_tags and record.get("record_type") == "episode":
            reflection = record.get("reflection_written") or record.get("reflection") or {}
            task_tags = list(reflection.get("task_tags", []))
        metadata = _task_tags_to_metadata(task_tags)
        if metadata:
            metadata_by_task[task_id] = metadata
    return metadata_by_task


def _summary_for_tasks(records: list[dict[str, Any]], task_ids: set[str]) -> dict[str, float]:
    filtered_records = [record for record in records if record.get("task_id") in task_ids]
    return _summary(filtered_records)


def _mode_step_and_episode_maps(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    steps_by_task: dict[str, list[dict[str, Any]]] = {}
    for step in _step_records(records):
        steps_by_task.setdefault(step["task_id"], []).append(step)
    episodes = _episode_records(records)
    return steps_by_task, episodes


def _process_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, list[float]]]:
    steps_by_task, episodes = _mode_step_and_episode_maps(records)
    ask_counts: list[float] = []
    choose_without_asking: list[float] = []
    reflection_admitted: list[float] = []

    for episode in episodes:
        task_steps = steps_by_task.get(episode["task_id"], [])
        asks = sum(1 for step in task_steps if str(step.get("action", "")).startswith("ASK"))
        ask_counts.append(float(asks))
        choose_without_asking.append(float(asks == 0))
        reflection_admitted.append(float(bool(episode.get("reflection_admitted"))))

    if not episodes:
        summary = {
            "avg_asks_per_episode": 0.0,
            "choose_without_asking_rate": 0.0,
            "reflection_admission_rate": 0.0,
        }
    else:
        episode_count = len(episodes)
        summary = {
            "avg_asks_per_episode": round(sum(ask_counts) / episode_count, 4),
            "choose_without_asking_rate": round(sum(choose_without_asking) / episode_count, 4),
            "reflection_admission_rate": round(sum(reflection_admitted) / episode_count, 4),
        }

    series = {
        "ask_counts": ask_counts,
        "choose_without_asking": choose_without_asking,
        "reflection_admitted": reflection_admitted,
    }
    return summary, series


def _subject_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[float]]]:
    steps = _step_records(records)
    episodes = _episode_records(records)
    ask_steps = [step for step in steps if str(step.get("action", "")).startswith("ASK")]
    endogenous_counts = {"epistemic": 0, "recognition": 0, "commitment": 0}
    other_dependence_values: list[float] = []
    deferred_samples: list[float] = []
    identity_continuity_values: list[float] = []
    commitment_scores: list[float] = []
    bearing_gaps: list[float] = []

    for step in steps:
        recognition_judgment = step.get("recognition_judgment") or {}
        lack_snapshot = step.get("lack_snapshot") or {}
        thin_self_model = step.get("thin_self_model") or {}
        action = str(step.get("action", ""))
        dominant_gap = str(lack_snapshot.get("dominant_gap", ""))
        choose_readiness = float(thin_self_model.get("choose_readiness", 0.0))

        if action.startswith("ASK") and dominant_gap in endogenous_counts:
            endogenous_counts[dominant_gap] += 1
        if recognition_judgment:
            other_dependence_values.append(float(recognition_judgment.get("other_dependence", 0.0)))
        if action.startswith("ASK"):
            deferred_samples.append(
                float(dominant_gap in {"epistemic", "recognition"} or choose_readiness < 0.55)
            )
        elif action.startswith("CHOOSE"):
            deferred_samples.append(float(choose_readiness >= 0.55 and dominant_gap != "epistemic"))

    for episode in episodes:
        arc_state = episode.get("arc_state") or {}
        commitment_record = episode.get("commitment_record") or {}
        if arc_state:
            identity_continuity_values.append(float(arc_state.get("identity_continuity_signal", 0.0)))
        if commitment_record:
            commitment_scores.append(float(commitment_record.get("attribution_score", 0.0)))
            bearing_gaps.append(1.0 - float(commitment_record.get("bearing_alignment", 0.0)))

    summary = {
        "endogenous_question_rate": {
            "overall": round(_safe_div(len(ask_steps), len(steps)), 4),
            "epistemic": round(_safe_div(endogenous_counts["epistemic"], len(ask_steps)), 4),
            "recognition": round(_safe_div(endogenous_counts["recognition"], len(ask_steps)), 4),
            "commitment": round(_safe_div(endogenous_counts["commitment"], len(ask_steps)), 4),
        },
        "other_dependence_index": round(_mean(other_dependence_values), 4),
        "identity_continuity_score": round(_mean(identity_continuity_values), 4),
        "i_saying_i_bearing_gap": round(_mean(bearing_gaps), 4),
        "commitment_attribution_score": round(_mean(commitment_scores), 4),
        "deferred_closure_efficiency": round(_mean(deferred_samples), 4),
    }
    series = {
        "other_dependence": other_dependence_values,
        "identity_continuity": identity_continuity_values,
        "commitment_attribution": commitment_scores,
        "bearing_gap": bearing_gaps,
        "deferred_closure": deferred_samples,
    }
    return summary, series


def _subgroup_summary(mode_records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    subgroup_summary: dict[str, dict[str, dict[str, Any]]] = {
        "difficulty": {},
        "noise_level": {},
    }
    metadata_by_mode = {
        mode_name: _task_metadata(records)
        for mode_name, records in mode_records.items()
    }

    for group_key, metadata_field in (("difficulty", "difficulty"), ("noise_level", "noise_level")):
        subgroup_values = sorted(
            {
                (
                    f"{metadata[metadata_field]:.2f}"
                    if isinstance(metadata.get(metadata_field), float)
                    else str(metadata.get(metadata_field))
                )
                for metadata_by_task in metadata_by_mode.values()
                for metadata in metadata_by_task.values()
                if metadata_field in metadata
            }
        )

        for subgroup_value in subgroup_values:
            subgroup_summary[group_key][subgroup_value] = {}
            for mode_name, records in mode_records.items():
                matching_task_ids = {
                    task_id
                    for task_id, metadata in metadata_by_mode[mode_name].items()
                    if (
                        f"{metadata[metadata_field]:.2f}"
                        if isinstance(metadata.get(metadata_field), float)
                        else str(metadata.get(metadata_field))
                    )
                    == subgroup_value
                }
                subgroup_summary[group_key][subgroup_value][mode_name] = _summary_for_tasks(
                    records,
                    matching_task_ids,
                )

    return subgroup_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute experiment summaries and mechanism metrics.")
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--perm", type=Path, required=True)
    parser.add_argument("--no-z", dest="no_self_model", type=Path, default=None)
    parser.add_argument("--no-cf", dest="no_cf_eval", type=Path, default=None)
    parser.add_argument("--selector-penalty", dest="selector_penalty", type=Path, default=None)
    parser.add_argument("--voi-coverage-only", dest="voi_coverage_only", type=Path, default=None)
    parser.add_argument("--voi-gate-only", dest="voi_gate_only", type=Path, default=None)
    parser.add_argument("--voi-memory-v1", dest="voi_memory_v1", type=Path, default=None)
    parser.add_argument("--subject-v0", dest="subject_v0", type=Path, default=None)
    parser.add_argument("--subject-no-other", dest="subject_no_other", type=Path, default=None)
    parser.add_argument("--subject-no-commitment", dest="subject_no_commitment", type=Path, default=None)
    parser.add_argument(
        "--subject-no-memory-recompose",
        dest="subject_no_memory_recompose",
        type=Path,
        default=None,
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode_records: dict[str, list[dict[str, Any]]] = {
        "full": _load_records(args.full),
        "ctrl": _load_records(args.ctrl),
        "perm": _load_records(args.perm),
    }
    for mode_name in SUPPLEMENTARY_MODES + EXPERIMENTAL_MODES + SUBJECT_MODES:
        path = getattr(args, mode_name, None)
        if path and path.exists():
            mode_records[mode_name] = _load_records(path)

    dcr, dcr_by_task = decision_change_rates(mode_records["full"], mode_records["ctrl"])
    cur, cur_by_task = decision_change_rates(mode_records["full"], mode_records["perm"])
    sri, sri_contributions = self_reinforcement_index(mode_records["full"])

    summary = {
        "full": _summary(mode_records["full"]),
        "ctrl": _summary(mode_records["ctrl"]),
        "perm": _summary(mode_records["perm"]),
        "DCR": round(dcr, 4),
        "CUR": round(cur, 4),
        "SRI": round(sri, 4),
    }
    supplementary_summary = {
        mode_name: _summary(records)
        for mode_name, records in mode_records.items()
        if mode_name in SUPPLEMENTARY_MODES
    }
    experimental_summary = {
        mode_name: _summary(records)
        for mode_name, records in mode_records.items()
        if mode_name in EXPERIMENTAL_MODES
    }

    process_summary: dict[str, dict[str, float]] = {}
    process_series: dict[str, list[float]] = {}
    metric_series: dict[str, list[float]] = {}
    subject_summary: dict[str, dict[str, Any]] = {}
    subject_series: dict[str, list[float]] = {}

    for mode_name, records in mode_records.items():
        process_metrics, mode_process_series = _process_metrics(records)
        process_summary[mode_name] = process_metrics
        metric_series[f"{mode_name}_rewards"] = _episode_series(records, "total_reward")
        metric_series[f"{mode_name}_steps"] = _episode_series(records, "steps")
        metric_series[f"{mode_name}_success"] = _episode_series(records, "success")
        process_series[f"{mode_name}_ask_counts"] = mode_process_series["ask_counts"]
        process_series[f"{mode_name}_choose_without_asking"] = mode_process_series["choose_without_asking"]
        process_series[f"{mode_name}_reflection_admitted"] = mode_process_series["reflection_admitted"]

        if mode_name in SUBJECT_MODES:
            mode_subject_summary, mode_subject_series = _subject_metrics(records)
            subject_summary[mode_name] = mode_subject_summary
            for key, values in mode_subject_series.items():
                subject_series[f"{mode_name}_{key}"] = values

    metrics = {
        "summary": summary,
        "supplementary_summary": supplementary_summary,
        "experimental_summary": experimental_summary,
        "subject_summary": subject_summary,
        "process_summary": process_summary,
        "subgroup_summary": _subgroup_summary(
            {
                mode_name: mode_records[mode_name]
                for mode_name in MAIN_MODES
            }
        ),
        "series": {
            **metric_series,
            **process_series,
            **subject_series,
            "dcr_by_task": dcr_by_task,
            "cur_by_task": cur_by_task,
            "sri_contributions": sri_contributions,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
