from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from agent.candidate_coverage import (
    VERIFY_PRUNED_REASONS,
    VERIFY_PRUNED_STAGES,
    best_family_surface_shape as coverage_best_family_surface_shape,
    build_candidate_layer_bundle,
    build_verify_lineage_survival,
    dedupe_candidates,
    infer_best_verify_missing_reason,
    infer_verify_candidate_exposed,
)
from agent.counterfactual import evaluate_candidate
from agent.voi_memory import evaluate_ask_gate
from env.toytextmdp import ToyTextMDP, load_tasks

MODE_SPECS = {
    "full": {
        "records": "full.jsonl",
        "memory": "full_memory.jsonl",
        "label": "full",
    },
    "no_reflection": {
        "records": "no_ref.jsonl",
        "memory": "no_ref_memory.jsonl",
        "label": "no_reflection",
    },
    "permute": {
        "records": "permute.jsonl",
        "memory": "permute_memory.jsonl",
        "label": "permute",
    },
    "no_self_model": {
        "records": "no_z.jsonl",
        "memory": "no_z_memory.jsonl",
        "label": "no_self_model",
    },
    "no_cf_eval": {
        "records": "no_cf.jsonl",
        "memory": "no_cf_memory.jsonl",
        "label": "no_cf_eval",
    },
    "selector_penalty": {
        "records": "selector_penalty.jsonl",
        "memory": "selector_penalty_memory.jsonl",
        "label": "selector_penalty",
    },
    "voi_coverage_only": {
        "records": "voi_coverage_only.jsonl",
        "memory": "voi_coverage_only_memory.jsonl",
        "label": "voi_coverage_only",
    },
    "voi_gate_only": {
        "records": "voi_gate_only.jsonl",
        "memory": "voi_gate_only_memory.jsonl",
        "label": "voi_gate_only",
    },
    "voi_memory_v1": {
        "records": "voi_memory_v1.jsonl",
        "memory": "voi_memory_v1_memory.jsonl",
        "label": "voi_memory_v1",
    },
}
EXPERIMENTAL_MODE_ORDER = (
    "selector_penalty",
    "voi_coverage_only",
    "voi_gate_only",
    "voi_memory_v1",
)
REFLECTION_FIELDS = ("info_seeking", "cost_sensitivity", "confidence_threshold")
VOI_GATE_MODES = {"voi_memory_v1", "voi_gate_only"}
PRE_SELECTOR_PRIMARY_EVIDENCE_FIELDS = (
    "verify_pruned_stage",
    "verify_pruned_reason",
    "verify_retained_before_selector",
    "verify_entered_selector",
    "ask_gate_enabled",
    "ask_gate_allowed",
    "ask_gate_reason",
    "pre_selector_block_source",
    "pre_selector_block_reason",
)
PRE_SELECTOR_SECONDARY_CONTEXT_FIELDS = (
    "ask_gate_score",
    "ask_advantage",
    "best_ask_family",
    "best_ask_eval",
    "best_choose_eval",
    "voi_ratio",
    "uncertainty",
    "base_uncertainty",
    "reliability_uncertainty",
    "evidence_conflict",
    "memory_signal",
    "helpful_ask",
    "choose_regret",
    "regretful_ask",
    "choose_helpful",
)
PRE_SELECTOR_CLAIM_BOUNDARY = (
    "This audit attributes which gate-facing reason accounted for the block; "
    "it does not recommend changing the gate."
)
BLOCKED_STEP_AUDIT_CLAIM_BOUNDARY = (
    "This audit classifies blocked strong_negative_voi steps into task pressure, "
    "surfaced ask deficit, or hard cutoff candidates; it does not recommend "
    "changing the gate."
)
BLOCKED_STEP_CLASS_ORDER = (
    "task_pressure",
    "surfaced_ask_deficit",
    "hard_cutoff_candidate",
)
BLOCKED_STEP_ROUTE_THRESHOLD = 0.60
BLOCKED_STEP_CLASS_LABELS = {
    "task_pressure": "task_pressure",
    "surfaced_ask_deficit": "candidate_coverage/runtime_backend",
    "hard_cutoff_candidate": "ask_gate_calibration",
    "mixed": "mixed",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
    if candidate.exists():
        return candidate
    return None


def _load_task_index_for_records(
    records_path: Path,
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tasks_path = _resolve_tasks_path(records_path, records)
    if tasks_path is None or not tasks_path.exists():
        return {}
    return {
        str(task.get("task_id")): task
        for task in load_tasks(tasks_path)
        if task.get("task_id")
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _round_optional(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    if x_var == 0 or y_var == 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(x_var * y_var)


def _parse_explicit_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def _detect_seeds(run_root: Path, raw_seeds: str | None) -> tuple[list[int], dict[int, Path]]:
    if raw_seeds:
        seeds = _parse_explicit_seeds(raw_seeds)
        return seeds, {seed: run_root / f"seed_{seed}" for seed in seeds}

    seed_dirs = [
        path
        for path in run_root.iterdir()
        if path.is_dir() and path.name.startswith("seed_")
    ]
    if seed_dirs:
        seeds = sorted(int(path.name.split("_", 1)[1]) for path in seed_dirs)
        return seeds, {seed: run_root / f"seed_{seed}" for seed in seeds}

    return [0], {0: run_root}


def _task_index(task_id: str) -> int:
    try:
        return int(str(task_id).split("-")[-1])
    except ValueError:
        return 0


def _noise_label(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _action_type(action: str) -> str:
    if str(action).startswith("ASK"):
        return "ASK"
    if str(action).startswith("CHOOSE"):
        return "CHOOSE"
    return "OTHER"


def _tool_family(action: str) -> str:
    action = str(action)
    if action.startswith("ASK_SCAN_"):
        return "scan"
    if action.startswith("ASK_VERIFY_"):
        return "verify"
    if action.startswith("ASK_CLUE_"):
        return "clue"
    if action.startswith("CHOOSE_"):
        return "choose"
    return "other"


def _count_candidates_by_family(candidates: list[dict[str, Any]], family: str) -> int:
    return sum(
        1
        for candidate in candidates
        if _tool_family(str(candidate.get("actions", [""])[0])) == family
    )


def _candidate_covers_action(candidates: list[dict[str, Any]], action: str | None) -> bool:
    if not action:
        return False
    return any(str(candidate.get("actions", [""])[0]) == action for candidate in candidates)


def _surface_shape_from_candidate(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "none"
    actions = [str(action) for action in candidate.get("actions", [])]
    category = str(candidate.get("category", ""))
    if not actions:
        if category == "ask_only":
            return "ask_only"
        if category == "ask_then_choose_fixed":
            return "fixed_choose"
        if category == "ask_then_choose":
            return "canonical_choose_best"
        return "none"
    first_action = actions[0]
    if _tool_family(first_action) not in {"scan", "verify"}:
        return "none"
    if len(actions) == 1:
        return "ask_only"
    if len(actions) == 2 and actions[1] == "CHOOSE_BEST":
        return "canonical_choose_best"
    if len(actions) == 2 and actions[1].startswith("CHOOSE_") and actions[1] != "CHOOSE_BEST":
        return "fixed_choose"
    return "multi_step"


def _surface_shape_from_row(row: dict[str, Any] | None) -> str:
    if not row:
        return "none"
    candidate = {
        "actions": list(row.get("actions", [])),
        "category": row.get("category"),
    }
    return _surface_shape_from_candidate(candidate)


def _candidate_action_keys(candidates: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {
        tuple(str(action) for action in candidate.get("actions", []))
        for candidate in candidates
        if candidate.get("actions")
    }


def _family_first_actions(candidates: list[dict[str, Any]], family: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        actions = [str(action) for action in candidate.get("actions", [])]
        if not actions or _tool_family(actions[0]) != family or actions[0] in seen:
            continue
        seen.add(actions[0])
        ordered.append(actions[0])
    return ordered


def _infer_pre_selector_filter_evidence(
    *,
    ask_gate_enabled: bool,
    ask_gate_allowed: bool | None,
    after_candidates: list[dict[str, Any]],
    selector_candidates: list[dict[str, Any]],
    logged_applied: Any = None,
    logged_empty: Any = None,
) -> dict[str, Any]:
    if logged_applied is not None or logged_empty is not None:
        applied = bool(logged_applied)
        empty = bool(logged_empty)
    elif ask_gate_enabled and ask_gate_allowed is False:
        filtered_pairs = [
            candidate
            for candidate in after_candidates
            if not str(candidate.get("actions", [""])[0]).startswith("ASK")
        ]
        if filtered_pairs:
            applied = _candidate_action_keys(selector_candidates) == _candidate_action_keys(filtered_pairs)
            empty = False
        else:
            applied = False
            empty = _candidate_action_keys(selector_candidates) == _candidate_action_keys(after_candidates)
    else:
        applied = False
        empty = False
    return {
        "pre_selector_filter_applied": applied,
        "pre_selector_filtered_pairs_empty": empty,
        "pre_selector_has_direct_evidence": bool(applied or empty),
    }


def _classify_pre_selector_block(
    *,
    ask_gate_enabled: bool,
    ask_gate_allowed: bool | None,
    ask_gate_reason: str | None,
    pre_selector_has_direct_evidence: bool,
    pre_selector_filtered_pairs_empty: bool,
) -> tuple[str | None, str | None]:
    if not pre_selector_has_direct_evidence:
        return "unknown", "unknown"
    if ask_gate_enabled and ask_gate_allowed is False:
        reason = str(ask_gate_reason or "")
        if reason in {"strong_negative_voi", "borderline_block"}:
            return "ask_gate", reason
        if reason == "blocked_fallback_original" and pre_selector_filtered_pairs_empty:
            return "ask_gate", "blocked_fallback_original"
        return "ask_gate", "unknown"
    return "other_pre_selector_filter", "unknown"


def _classify_oracle_mismatch(
    *,
    selected_type: str,
    selected_tool_family: str,
    tool_choice_applicable: bool,
    tool_optimal_family: str | None,
    oracle_tool_choice_applicable: bool,
    oracle_tool_optimal_family: str | None,
) -> tuple[bool, str, str]:
    if not oracle_tool_choice_applicable or oracle_tool_optimal_family not in {"scan", "verify"}:
        return False, "none", "none"

    if selected_type == "CHOOSE":
        return True, "choose_swallow", "none"

    mismatch = selected_tool_family != oracle_tool_optimal_family
    mismatch_kind = "tool_family_mismatch" if mismatch else "none"
    if not mismatch or not tool_choice_applicable or tool_optimal_family in {None, "tie"}:
        return mismatch, mismatch_kind, "none"

    if (
        tool_optimal_family != oracle_tool_optimal_family
        and selected_tool_family != oracle_tool_optimal_family
        and selected_tool_family != tool_optimal_family
    ):
        return mismatch, mismatch_kind, "mixed_surface_and_policy_error"
    if selected_tool_family == tool_optimal_family and tool_optimal_family != oracle_tool_optimal_family:
        return mismatch, mismatch_kind, "surface_family_distortion"
    if tool_optimal_family == oracle_tool_optimal_family and selected_tool_family != oracle_tool_optimal_family:
        return mismatch, mismatch_kind, "policy_family_error"
    return mismatch, mismatch_kind, "none"


def _family_label_from_scores(
    left_score: float | None,
    right_score: float | None,
    *,
    left_family: str,
    right_family: str,
) -> str | None:
    if left_score is None or right_score is None:
        return None
    if math.isclose(left_score, right_score, rel_tol=1e-9, abs_tol=1e-9):
        return "tie"
    return left_family if left_score > right_score else right_family


def _replay_env_to_step(
    *,
    task_config: dict[str, Any],
    prior_actions: list[str],
) -> tuple[ToyTextMDP, dict[str, Any]] | None:
    env = ToyTextMDP(task_config)
    observation, _ = env.reset()
    for action in prior_actions:
        observation, _, done, _ = env.step(action)
        if done:
            return None
    return env, observation


def _best_oracle_candidate_for_actions(
    env: ToyTextMDP,
    *,
    actions: list[str],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for action in actions:
        candidate = {
            "name": f"{action.lower()}_then_choose_best",
            "actions": [action, "CHOOSE_BEST"],
        }
        evaluation = evaluate_candidate(env, candidate)
        sim_env = env.clone()
        observation, _, _, _ = sim_env.step(action)
        row = {
            "action": action,
            "eval_score": float(evaluation.get("score", 0.0)),
            "observation_text": str(observation.get("text", "")),
        }
        if best is None or (row["eval_score"], row["action"]) > (best["eval_score"], best["action"]):
            best = row
    return best


def _best_surface_eval_for_action_type(
    step_record: dict[str, Any],
    *,
    action_type: str,
) -> dict[str, Any] | None:
    evaluation_by_name = {
        str(evaluation.get("candidate_name")): evaluation
        for evaluation in step_record.get("evaluations", [])
        if evaluation.get("candidate_name") is not None
    }
    best: dict[str, Any] | None = None
    for candidate in step_record.get("candidates", []):
        actions = [str(action) for action in candidate.get("actions", [])]
        if not actions:
            continue
        first_action = actions[0]
        if action_type == "ASK" and not first_action.startswith("ASK"):
            continue
        if action_type == "CHOOSE" and not first_action.startswith("CHOOSE"):
            continue
        evaluation = evaluation_by_name.get(str(candidate.get("name")))
        if evaluation is None:
            continue
        row = {
            "candidate_name": str(candidate.get("name")),
            "action": first_action,
            "eval_score": float(evaluation.get("score", 0.0)),
        }
        if best is None or (row["eval_score"], row["action"]) > (best["eval_score"], best["action"]):
            best = row

    ask_gate_inputs = dict(step_record.get("ask_gate_inputs") or {})
    if best is None and action_type == "ASK" and ask_gate_inputs.get("best_ask_eval") is not None:
        return {
            "candidate_name": None,
            "action": None,
            "eval_score": float(ask_gate_inputs.get("best_ask_eval", 0.0)),
        }
    if best is None and action_type == "CHOOSE" and ask_gate_inputs.get("best_choose_eval") is not None:
        return {
            "candidate_name": None,
            "action": None,
            "eval_score": float(ask_gate_inputs.get("best_choose_eval", 0.0)),
        }
    return best


def _classify_blocked_step(
    *,
    oracle_best_ask_eval: float | None,
    best_choose_eval: float | None,
    best_surfaced_ask_eval: float | None,
    tolerance: float = 1e-6,
) -> str:
    if (
        oracle_best_ask_eval is None
        or best_choose_eval is None
        or oracle_best_ask_eval <= best_choose_eval + tolerance
    ):
        return "task_pressure"
    if best_surfaced_ask_eval is None or best_surfaced_ask_eval <= best_choose_eval + tolerance:
        return "surfaced_ask_deficit"
    return "hard_cutoff_candidate"


def _matches_avoid(action: str, avoid_action: str) -> bool:
    if avoid_action in {"ASK", "CHOOSE"}:
        return action.startswith(avoid_action)
    return action == avoid_action


def _suppresses_choose(avoid_action: str) -> bool:
    return avoid_action == "CHOOSE" or str(avoid_action).startswith("CHOOSE_")


def _suppresses_ask(avoid_action: str) -> bool:
    return avoid_action == "ASK" or str(avoid_action).startswith("ASK_")


def _task_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for tag in record.get("task_tags", []):
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
        elif key == "tool_profile":
            metadata["tool_profile"] = raw_value
        elif key == "tool_subtype":
            metadata["tool_subtype"] = raw_value
    return metadata


def _base_candidates_from_step_record(step_record: dict[str, Any]) -> list[dict[str, Any]]:
    logged_before = list(step_record.get("candidates_before_normalization") or [])
    if logged_before:
        return dedupe_candidates(logged_before)

    template_candidates = list(step_record.get("template_candidates") or [])
    llm_candidates = list(step_record.get("llm_candidates") or [])
    final_candidates = list(step_record.get("candidates") or [])
    candidate_backend = str(step_record.get("candidate_backend", "unknown"))

    if candidate_backend == "openai_responses_plus_template":
        return dedupe_candidates(llm_candidates + template_candidates)
    if candidate_backend == "openai_responses":
        return dedupe_candidates(llm_candidates or final_candidates)
    return dedupe_candidates(template_candidates or final_candidates)


def _evaluate_candidates_for_env(
    env: ToyTextMDP | None,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if env is None:
        return []
    return [evaluate_candidate(env, candidate) for candidate in candidates]


def _candidate_layer_snapshot(
    *,
    step_record: dict[str, Any],
    mode_name: str,
    belief: dict[str, Any],
    self_model: dict[str, Any],
    task_config: dict[str, Any] | None = None,
    prior_actions: list[str] | None = None,
) -> dict[str, Any]:
    template_candidates = list(step_record.get("template_candidates") or [])
    candidate_limit = int(step_record.get("candidate_limit", 5) or 5)
    fallback_bundle = build_candidate_layer_bundle(
        base_candidates=_base_candidates_from_step_record(step_record),
        available_actions=[str(action) for action in belief.get("remaining_asks", [])],
        template_candidates=template_candidates,
    )
    has_logged_candidate_layer = "candidate_count_by_family_after_normalization" in step_record
    if has_logged_candidate_layer:
        before_candidates = dedupe_candidates(list(step_record.get("candidates_before_normalization") or []))
        after_candidates = dedupe_candidates(list(step_record.get("candidates") or []))
        before_evaluations = list(step_record.get("evaluations_before_normalization") or [])
        after_evaluations = list(step_record.get("evaluations") or [])
        selector_candidates = list(step_record.get("selector_candidates") or after_candidates)
        selector_evaluations = list(step_record.get("selector_evaluations") or after_evaluations)
        before_counts = dict(step_record.get("candidate_count_by_family_before_normalization") or {})
        after_counts = dict(step_record.get("candidate_count_by_family_after_normalization") or {})
        source_candidates = dedupe_candidates(
            list(step_record.get("source_candidates") or fallback_bundle["source_candidates"])
        )
        normalized_candidates_pre_dedup = list(
            step_record.get("normalized_candidates_pre_dedup") or fallback_bundle["normalized_candidates_pre_dedup"]
        )
        deduped_candidates = dedupe_candidates(
            list(
                step_record.get("candidates_after_dedup_before_exposure")
                or fallback_bundle["candidates_after_dedup_before_exposure"]
            )
        )
    else:
        bundle = fallback_bundle
        before_candidates = list(bundle["candidates_before_normalization"])
        after_candidates = list(bundle["candidates_after_normalization"])
        before_counts = dict(bundle["candidate_count_by_family_before_normalization"])
        after_counts = dict(bundle["candidate_count_by_family_after_normalization"])
        source_candidates = list(bundle["source_candidates"])
        normalized_candidates_pre_dedup = list(bundle["normalized_candidates_pre_dedup"])
        deduped_candidates = list(bundle["candidates_after_dedup_before_exposure"])

        replay = None
        if task_config is not None:
            replay = _replay_env_to_step(
                task_config=task_config,
                prior_actions=list(prior_actions or []),
            )
        replay_env = replay[0] if replay is not None else None
        before_evaluations = _evaluate_candidates_for_env(replay_env, before_candidates)
        after_evaluations = _evaluate_candidates_for_env(replay_env, after_candidates)
        if not before_evaluations:
            before_evaluations = list(step_record.get("evaluations_before_normalization") or [])
        if not after_evaluations:
            after_evaluations = list(step_record.get("evaluations") or [])

        selector_candidates = after_candidates
        selector_evaluations = after_evaluations
        if mode_name in VOI_GATE_MODES and after_evaluations:
            ask_gate = evaluate_ask_gate(
                candidates=after_candidates,
                evaluations=after_evaluations,
                belief=belief,
                confidence_threshold=float(self_model.get("confidence_threshold", 1.25)),
                ask_cost=float(
                    (task_config or {}).get(
                        "ask_cost",
                        ((step_record.get("ask_gate_inputs") or {}).get("ask_cost", -0.05)),
                    )
                ),
                memory_retrieval=dict(step_record.get("retrieved_decision_lessons") or {}),
            )
            if not ask_gate["allowed"]:
                filtered_pairs = [
                    (candidate, evaluation)
                    for candidate, evaluation in zip(after_candidates, after_evaluations, strict=True)
                    if not str(candidate.get("actions", [""])[0]).startswith("ASK")
                ]
                if filtered_pairs:
                    selector_candidates = [candidate for candidate, _ in filtered_pairs]
                    selector_evaluations = [evaluation for _, evaluation in filtered_pairs]

    best_scan_surface_shape_before = str(step_record.get("best_scan_surface_shape_before") or "")
    if not best_scan_surface_shape_before:
        best_scan_surface_shape_before = coverage_best_family_surface_shape(
            before_candidates,
            before_evaluations,
            "scan",
        )
    best_verify_surface_shape_before = str(step_record.get("best_verify_surface_shape_before") or "")
    if not best_verify_surface_shape_before:
        best_verify_surface_shape_before = coverage_best_family_surface_shape(
            before_candidates,
            before_evaluations,
            "verify",
        )
    best_scan_surface_shape_after = str(step_record.get("best_scan_surface_shape_after") or "")
    if not best_scan_surface_shape_after:
        best_scan_surface_shape_after = coverage_best_family_surface_shape(
            after_candidates,
            after_evaluations,
            "scan",
        )
    best_verify_surface_shape_after = str(step_record.get("best_verify_surface_shape_after") or "")
    if not best_verify_surface_shape_after:
        best_verify_surface_shape_after = coverage_best_family_surface_shape(
            after_candidates,
            after_evaluations,
            "verify",
        )
    verify_candidate_exposed = bool(
        step_record.get("verify_candidate_exposed")
        if "verify_candidate_exposed" in step_record
        else infer_verify_candidate_exposed(selector_candidates)
    )
    best_verify_missing_reason = str(step_record.get("best_verify_missing_reason") or "")
    if not best_verify_missing_reason:
        best_verify_missing_reason = infer_best_verify_missing_reason(
            remaining_verify_actions=[str(action) for action in belief.get("remaining_verify_actions", [])],
            candidates_before_normalization=before_candidates,
            candidates_after_normalization=after_candidates,
            selector_candidates=selector_candidates,
            selector_evaluations=selector_evaluations,
        )
    pre_selector_evidence = _infer_pre_selector_filter_evidence(
        ask_gate_enabled=bool(step_record.get("ask_gate_allowed") is not None),
        ask_gate_allowed=step_record.get("ask_gate_allowed"),
        after_candidates=after_candidates,
        selector_candidates=selector_candidates,
        logged_applied=step_record.get("pre_selector_filter_applied"),
        logged_empty=step_record.get("pre_selector_filtered_pairs_empty"),
    )
    verify_lineage_audit = list(step_record.get("verify_lineage_audit") or [])
    if not verify_lineage_audit:
        verify_lineage_audit = build_verify_lineage_survival(
            remaining_verify_actions=[str(action) for action in belief.get("remaining_verify_actions", [])],
            candidate_backend=str(step_record.get("candidate_backend", "unknown")),
            candidate_limit=candidate_limit,
            template_candidates=template_candidates,
            llm_candidates=list(step_record.get("llm_candidates") or []),
            source_candidates=source_candidates,
            normalized_candidates_pre_dedup=normalized_candidates_pre_dedup,
            deduped_candidates=deduped_candidates,
            final_candidates=after_candidates,
            selector_candidates=selector_candidates,
            ask_gate_allowed=step_record.get("ask_gate_allowed"),
            pre_selector_filter_evidence=pre_selector_evidence["pre_selector_has_direct_evidence"],
        )

    return {
        "source_candidates": source_candidates,
        "before_candidates": before_candidates,
        "before_evaluations": before_evaluations,
        "after_candidates": after_candidates,
        "after_evaluations": after_evaluations,
        "normalized_candidates_pre_dedup": normalized_candidates_pre_dedup,
        "candidates_after_dedup_before_exposure": deduped_candidates,
        "selector_candidates_after_candidate_layer": selector_candidates,
        "selector_evaluations_after_candidate_layer": selector_evaluations,
        "candidate_count_by_family_before_normalization": before_counts,
        "candidate_count_by_family_after_normalization": after_counts,
        "best_scan_surface_shape_before": best_scan_surface_shape_before,
        "best_verify_surface_shape_before": best_verify_surface_shape_before,
        "best_scan_surface_shape_after": best_scan_surface_shape_after,
        "best_verify_surface_shape_after": best_verify_surface_shape_after,
        "verify_candidate_exposed": verify_candidate_exposed,
        "best_verify_missing_reason": best_verify_missing_reason,
        "verify_lineage_audit": verify_lineage_audit,
        **pre_selector_evidence,
    }


def _load_memory_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    entries = _load_jsonl(path)
    return {
        str(entry.get("memory_id")): entry
        for entry in entries
        if entry.get("memory_id")
    }


def _memory_penalty(
    action: str,
    retrieved_entries: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    penalty = 0.0
    matched_ids: list[str] = []
    for entry in retrieved_entries:
        for avoid_action in entry.get("avoid_actions", []):
            if _matches_avoid(action, str(avoid_action)):
                penalty -= 1.0
                matched_ids.append(str(entry.get("memory_id", "unknown")))
                break
    return penalty, matched_ids


def _preference_score(
    action: str,
    belief: dict[str, Any],
    self_model: dict[str, Any],
) -> float:
    margin = float(belief.get("margin", 0.0))
    info_seeking = float(self_model.get("info_seeking", 0.0))
    cost_sensitivity = float(self_model.get("cost_sensitivity", 0.0))
    confidence_threshold = float(self_model.get("confidence_threshold", 0.0))

    if action.startswith("ASK"):
        return 0.9 * info_seeking - 0.5 * cost_sensitivity

    chosen = action.removeprefix("CHOOSE_")
    score = 0.8 * cost_sensitivity
    if chosen == belief.get("top_choice"):
        score += 0.3
    else:
        score -= 0.4

    if margin < confidence_threshold:
        score -= confidence_threshold - margin
    else:
        score += 0.2 * margin
    return score


def _candidate_rows_from_inputs(
    *,
    candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    retrieved_entries: list[dict[str, Any]],
    belief: dict[str, Any],
    self_model: dict[str, Any],
) -> list[dict[str, Any]]:
    evaluation_by_name = {
        evaluation["candidate_name"]: evaluation
        for evaluation in evaluations
    }
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate.get("name", ""))
        evaluation = evaluation_by_name.get(name, {})
        actions = [str(action) for action in candidate.get("actions", [])]
        first_action = str(actions[0]) if actions else ""
        action_type = _action_type(first_action)
        eval_score = float(evaluation.get("score", 0.0))
        preference = _preference_score(first_action, belief, self_model)
        memory_penalty, memory_hits = _memory_penalty(first_action, retrieved_entries)
        total = eval_score + 0.35 * preference + 0.6 * memory_penalty
        rows.append(
            {
                "name": name,
                "first_action": first_action,
                "actions": actions,
                "category": str(candidate.get("category", "")),
                "action_type": action_type,
                "eval_score": eval_score,
                "preference_score": preference,
                "memory_penalty": memory_penalty,
                "memory_hits": memory_hits,
                "total_score": total,
            }
        )
    return rows


def _candidate_rows(
    step_record: dict[str, Any],
    memory_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    retrieved_entries = [
        memory_index[memory_id]
        for memory_id in step_record.get("retrieved_memory_ids", [])
        if memory_id in memory_index
    ]
    belief = dict(step_record.get("belief", {}))
    self_model = dict(step_record.get("self_model", {}))
    rows = _candidate_rows_from_inputs(
        candidates=list(step_record.get("candidates", [])),
        evaluations=list(step_record.get("evaluations", [])),
        retrieved_entries=retrieved_entries,
        belief=belief,
        self_model=self_model,
    )
    return rows, retrieved_entries, belief, self_model


def _best_candidate(
    candidate_rows: list[dict[str, Any]],
    action_type: str,
    field_name: str,
) -> dict[str, Any] | None:
    filtered = [row for row in candidate_rows if row["action_type"] == action_type]
    if not filtered:
        return None
    return max(filtered, key=lambda row: (float(row[field_name]), row["name"]))


def _analyze_step_record(
    *,
    seed: int,
    mode_name: str,
    step_record: dict[str, Any],
    memory_index: dict[str, dict[str, Any]],
    task_config: dict[str, Any] | None = None,
    prior_actions: list[str] | None = None,
) -> dict[str, Any]:
    metadata = _task_metadata(step_record)
    candidate_rows, retrieved_entries, belief, self_model = _candidate_rows(step_record, memory_index)
    candidate_layer_snapshot = _candidate_layer_snapshot(
        step_record=step_record,
        mode_name=mode_name,
        belief=belief,
        self_model=self_model,
        task_config=task_config,
        prior_actions=prior_actions,
    )
    selected_action = str(step_record.get("action", ""))
    selected_type = _action_type(selected_action)
    margin = float(belief.get("margin", 0.0))
    confidence_threshold = float(self_model.get("confidence_threshold", 0.0))
    asked_count = int(belief.get("asked_count", 0))

    best_ask_eval = _best_candidate(candidate_rows, "ASK", "eval_score")
    best_choose_eval = _best_candidate(candidate_rows, "CHOOSE", "eval_score")
    best_ask_total = _best_candidate(candidate_rows, "ASK", "total_score")
    best_choose_total = _best_candidate(candidate_rows, "CHOOSE", "total_score")
    scan_rows = [row for row in candidate_rows if _tool_family(row["first_action"]) == "scan"]
    verify_rows = [row for row in candidate_rows if _tool_family(row["first_action"]) == "verify"]
    best_scan_eval = max(scan_rows, key=lambda row: (float(row["eval_score"]), row["name"])) if scan_rows else None
    best_verify_eval = max(verify_rows, key=lambda row: (float(row["eval_score"]), row["name"])) if verify_rows else None
    best_scan_surface_shape = _surface_shape_from_row(best_scan_eval)
    best_verify_surface_shape = _surface_shape_from_row(best_verify_eval)

    ask_advantage: float | None = None
    if best_ask_eval and best_choose_eval:
        ask_advantage = float(best_ask_eval["eval_score"]) - float(best_choose_eval["eval_score"])

    regretful_ask = bool(
        selected_type == "ASK"
        and best_ask_eval
        and best_choose_eval
        and float(best_choose_eval["eval_score"]) >= float(best_ask_eval["eval_score"])
    )
    choose_regret = bool(
        selected_type == "CHOOSE"
        and best_ask_eval
        and best_choose_eval
        and float(best_ask_eval["eval_score"]) > float(best_choose_eval["eval_score"])
    )

    selected_eval_score = float(step_record.get("selection_breakdown", {}).get("eval_score", 0.0))
    over_ask_cost = 0.0
    if selected_type == "ASK" and ask_advantage is not None and ask_advantage <= 0 and best_choose_eval:
        over_ask_cost = max(0.0, float(best_choose_eval["eval_score"]) - selected_eval_score)
    under_ask_cost = 0.0
    if selected_type == "CHOOSE" and ask_advantage is not None and ask_advantage > 0 and best_ask_eval:
        under_ask_cost = max(0.0, float(best_ask_eval["eval_score"]) - selected_eval_score)
    selected_tool_family = _tool_family(selected_action)
    tool_choice_applicable = bool(
        selected_tool_family in {"scan", "verify"}
        and best_scan_eval is not None
        and best_verify_eval is not None
    )
    tool_optimal_family: str | None = None
    if tool_choice_applicable:
        scan_score = float(best_scan_eval["eval_score"])
        verify_score = float(best_verify_eval["eval_score"])
        if math.isclose(scan_score, verify_score, rel_tol=1e-9, abs_tol=1e-9):
            tool_optimal_family = "tie"
        else:
            tool_optimal_family = "scan" if scan_score > verify_score else "verify"
    correct_tool_choice = bool(
        tool_choice_applicable
        and tool_optimal_family in {"tie", selected_tool_family}
    )
    wrong_tool_choice = bool(
        tool_choice_applicable
        and tool_optimal_family not in {None, "tie", selected_tool_family}
    )
    over_tool_cost = 0.0
    under_tool_cost = 0.0
    if tool_choice_applicable and selected_tool_family == "verify" and best_scan_eval:
        over_tool_cost = max(0.0, float(best_scan_eval["eval_score"]) - selected_eval_score)
    if tool_choice_applicable and selected_tool_family == "scan" and best_verify_eval:
        under_tool_cost = max(0.0, float(best_verify_eval["eval_score"]) - selected_eval_score)

    retrieved_memory_ids = [str(memory_id) for memory_id in step_record.get("retrieved_memory_ids", [])]
    choose_suppressing_memory_ids = [
        str(entry.get("memory_id"))
        for entry in retrieved_entries
        if any(_suppresses_choose(str(avoid_action)) for avoid_action in entry.get("avoid_actions", []))
    ]
    ask_suppressing_memory_ids = [
        str(entry.get("memory_id"))
        for entry in retrieved_entries
        if any(_suppresses_ask(str(avoid_action)) for avoid_action in entry.get("avoid_actions", []))
    ]
    retrieved_decision_lessons = dict(step_record.get("retrieved_decision_lessons") or {})
    positive_lessons = list(retrieved_decision_lessons.get("positive", []))
    caution_lessons = list(retrieved_decision_lessons.get("caution", []))
    retrieved_lesson_types = [
        str(entry.get("lesson_type", ""))
        for entry in positive_lessons + caution_lessons
        if entry.get("lesson_type")
    ]
    ask_gate_allowed = step_record.get("ask_gate_allowed")
    ask_gate_reason = str(step_record.get("ask_gate_reason", ""))
    ask_gate_enabled = ask_gate_allowed is not None
    ask_gate_inputs = dict(step_record.get("ask_gate_inputs") or {})

    selector_prefers_ask_nonpositive = bool(
        selected_type == "ASK"
        and ask_advantage is not None
        and ask_advantage <= 0
        and best_ask_total
        and best_choose_total
        and float(best_ask_total["total_score"]) > float(best_choose_total["total_score"])
    )

    best_scan_total = max(scan_rows, key=lambda row: (float(row["total_score"]), row["name"])) if scan_rows else None
    best_verify_total = max(verify_rows, key=lambda row: (float(row["total_score"]), row["name"])) if verify_rows else None
    selector_family_applicable = bool(best_scan_total is not None and best_verify_total is not None)
    selector_total_optimal_family = _family_label_from_scores(
        float(best_scan_total["total_score"]) if best_scan_total else None,
        float(best_verify_total["total_score"]) if best_verify_total else None,
        left_family="scan",
        right_family="verify",
    )
    surfaced_eval_optimal_family = _family_label_from_scores(
        float(best_scan_eval["eval_score"]) if best_scan_eval else None,
        float(best_verify_eval["eval_score"]) if best_verify_eval else None,
        left_family="scan",
        right_family="verify",
    )
    selector_family_override = bool(
        selector_family_applicable
        and surfaced_eval_optimal_family not in {None, "tie"}
        and selector_total_optimal_family not in {None, "tie"}
        and selector_total_optimal_family != surfaced_eval_optimal_family
    )

    template_candidates = list(step_record.get("template_candidates") or [])
    llm_candidates = list(step_record.get("llm_candidates") or [])
    final_candidates = list(step_record.get("candidates") or [])
    selector_candidates = list(step_record.get("selector_candidates") or final_candidates)
    candidate_backend = str(step_record.get("candidate_backend", "unknown"))
    if not template_candidates and candidate_backend in {"template", "template_fallback"}:
        template_candidates = final_candidates
    if not llm_candidates and candidate_backend == "openai_responses":
        llm_candidates = final_candidates

    remaining_scan_actions = [str(action) for action in belief.get("remaining_scan_actions", [])]
    remaining_verify_actions = [str(action) for action in belief.get("remaining_verify_actions", [])]

    oracle_best_scan: dict[str, Any] | None = None
    oracle_best_verify: dict[str, Any] | None = None
    oracle_tool_optimal_family: str | None = None
    oracle_tool_choice_applicable = False
    if task_config is not None:
        replay = _replay_env_to_step(
            task_config=task_config,
            prior_actions=list(prior_actions or []),
        )
        if replay is not None:
            oracle_env, _ = replay
            oracle_best_scan = _best_oracle_candidate_for_actions(
                oracle_env,
                actions=remaining_scan_actions,
            )
            oracle_best_verify = _best_oracle_candidate_for_actions(
                oracle_env,
                actions=remaining_verify_actions,
            )
            oracle_tool_choice_applicable = bool(oracle_best_scan and oracle_best_verify)
            oracle_tool_optimal_family = _family_label_from_scores(
                oracle_best_scan["eval_score"] if oracle_best_scan else None,
                oracle_best_verify["eval_score"] if oracle_best_verify else None,
                left_family="scan",
                right_family="verify",
            )

    final_oracle_scan_action_covered = _candidate_covers_action(
        final_candidates,
        oracle_best_scan["action"] if oracle_best_scan else None,
    )
    final_oracle_verify_action_covered = _candidate_covers_action(
        final_candidates,
        oracle_best_verify["action"] if oracle_best_verify else None,
    )
    template_oracle_scan_action_covered = _candidate_covers_action(
        template_candidates,
        oracle_best_scan["action"] if oracle_best_scan else None,
    )
    template_oracle_verify_action_covered = _candidate_covers_action(
        template_candidates,
        oracle_best_verify["action"] if oracle_best_verify else None,
    )
    llm_oracle_scan_action_covered = _candidate_covers_action(
        llm_candidates,
        oracle_best_scan["action"] if oracle_best_scan else None,
    )
    llm_oracle_verify_action_covered = _candidate_covers_action(
        llm_candidates,
        oracle_best_verify["action"] if oracle_best_verify else None,
    )
    selector_oracle_scan_action_covered = _candidate_covers_action(
        selector_candidates,
        oracle_best_scan["action"] if oracle_best_scan else None,
    )
    selector_oracle_verify_action_covered = _candidate_covers_action(
        selector_candidates,
        oracle_best_verify["action"] if oracle_best_verify else None,
    )

    final_oracle_family_alignment = bool(
        oracle_tool_choice_applicable
        and surfaced_eval_optimal_family in {"tie", oracle_tool_optimal_family}
    )
    selected_vs_oracle_family_mismatch, selected_vs_oracle_mismatch_kind, tool_family_error_source = (
        _classify_oracle_mismatch(
            selected_type=selected_type,
            selected_tool_family=selected_tool_family,
            tool_choice_applicable=tool_choice_applicable,
            tool_optimal_family=tool_optimal_family,
            oracle_tool_choice_applicable=oracle_tool_choice_applicable,
            oracle_tool_optimal_family=oracle_tool_optimal_family,
        )
    )
    final_scan_eval_gap = None
    if oracle_best_scan is not None and best_scan_eval is not None:
        final_scan_eval_gap = max(0.0, oracle_best_scan["eval_score"] - float(best_scan_eval["eval_score"]))
    final_verify_eval_gap = None
    if oracle_best_verify is not None and best_verify_eval is not None:
        final_verify_eval_gap = max(0.0, oracle_best_verify["eval_score"] - float(best_verify_eval["eval_score"]))
    verify_lineage_audit = list(candidate_layer_snapshot["verify_lineage_audit"])
    verify_lineage_generated_count = sum(int(row["verify_generated"]) for row in verify_lineage_audit)
    verify_lineage_before_selector_count = sum(
        int(row["verify_retained_before_selector"])
        for row in verify_lineage_audit
    )
    verify_lineage_entered_selector_count = sum(
        int(row["verify_entered_selector"])
        for row in verify_lineage_audit
    )
    verify_lineage_injected_before_selector_count = sum(
        int(row["verify_injected_before_selector"])
        for row in verify_lineage_audit
    )

    return {
        "seed": seed,
        "mode": mode_name,
        "task_id": str(step_record.get("task_id")),
        "task_index": _task_index(str(step_record.get("task_id"))),
        "step_index": int(step_record.get("step", 0)),
        "difficulty": str(metadata.get("difficulty", "unknown")),
        "noise_level": _noise_label(metadata.get("noise_level", "unknown")),
        "tool_profile": str(metadata.get("tool_profile", "none")),
        "tool_subtype": str(metadata.get("tool_subtype", "none")),
        "asked_count": asked_count,
        "selected_action": selected_action,
        "selected_type": selected_type,
        "selected_tool_family": selected_tool_family,
        "selected_eval_score": _round(selected_eval_score),
        "selected_total_score": _round(float(step_record.get("total_selection_score", 0.0))),
        "candidate_backend": candidate_backend,
        "ask_advantage": _round_optional(ask_advantage),
        "ask_gate_best_ask_family": str(ask_gate_inputs.get("best_ask_family") or "") or None,
        "ask_gate_best_ask_eval": _round_optional(ask_gate_inputs.get("best_ask_eval")),
        "ask_gate_best_choose_eval": _round_optional(ask_gate_inputs.get("best_choose_eval")),
        "ask_gate_voi_ratio": _round_optional(ask_gate_inputs.get("voi_ratio")),
        "ask_gate_uncertainty": _round_optional(ask_gate_inputs.get("uncertainty")),
        "ask_gate_base_uncertainty": _round_optional(ask_gate_inputs.get("base_uncertainty")),
        "ask_gate_reliability_uncertainty": _round_optional(
            ask_gate_inputs.get("reliability_uncertainty")
        ),
        "ask_gate_evidence_conflict": _round_optional(ask_gate_inputs.get("evidence_conflict")),
        "ask_gate_memory_signal": _round_optional(ask_gate_inputs.get("memory_signal")),
        "ask_gate_helpful_ask": _round_optional(ask_gate_inputs.get("helpful_ask")),
        "ask_gate_choose_regret": _round_optional(ask_gate_inputs.get("choose_regret")),
        "ask_gate_regretful_ask": _round_optional(ask_gate_inputs.get("regretful_ask")),
        "ask_gate_choose_helpful": _round_optional(ask_gate_inputs.get("choose_helpful")),
        "best_ask_eval_score": _round_optional(best_ask_eval["eval_score"] if best_ask_eval else None),
        "best_choose_eval_score": _round_optional(best_choose_eval["eval_score"] if best_choose_eval else None),
        "best_ask_total_score": _round_optional(best_ask_total["total_score"] if best_ask_total else None),
        "best_choose_total_score": _round_optional(best_choose_total["total_score"] if best_choose_total else None),
        "best_scan_eval_score": _round_optional(best_scan_eval["eval_score"] if best_scan_eval else None),
        "best_verify_eval_score": _round_optional(best_verify_eval["eval_score"] if best_verify_eval else None),
        "best_scan_surface_shape": best_scan_surface_shape,
        "best_verify_surface_shape": best_verify_surface_shape,
        "best_scan_surface_shape_before": candidate_layer_snapshot["best_scan_surface_shape_before"],
        "best_verify_surface_shape_before": candidate_layer_snapshot["best_verify_surface_shape_before"],
        "best_scan_surface_shape_after": candidate_layer_snapshot["best_scan_surface_shape_after"],
        "best_verify_surface_shape_after": candidate_layer_snapshot["best_verify_surface_shape_after"],
        "best_scan_total_score": _round_optional(best_scan_total["total_score"] if best_scan_total else None),
        "best_verify_total_score": _round_optional(best_verify_total["total_score"] if best_verify_total else None),
        "tool_choice_applicable": tool_choice_applicable,
        "tool_optimal_family": tool_optimal_family,
        "correct_tool_choice": correct_tool_choice,
        "wrong_tool_choice": wrong_tool_choice,
        "over_tool_cost": _round(over_tool_cost),
        "under_tool_cost": _round(under_tool_cost),
        "candidate_count_by_family_before_normalization": candidate_layer_snapshot[
            "candidate_count_by_family_before_normalization"
        ],
        "candidate_count_by_family_after_normalization": candidate_layer_snapshot[
            "candidate_count_by_family_after_normalization"
        ],
        "before_normalization_scan_candidate_count": int(
            candidate_layer_snapshot["candidate_count_by_family_before_normalization"].get("scan", 0)
        ),
        "before_normalization_verify_candidate_count": int(
            candidate_layer_snapshot["candidate_count_by_family_before_normalization"].get("verify", 0)
        ),
        "after_normalization_scan_candidate_count": int(
            candidate_layer_snapshot["candidate_count_by_family_after_normalization"].get("scan", 0)
        ),
        "after_normalization_verify_candidate_count": int(
            candidate_layer_snapshot["candidate_count_by_family_after_normalization"].get("verify", 0)
        ),
        "verify_candidate_exposed": bool(candidate_layer_snapshot["verify_candidate_exposed"]),
        "best_verify_missing_reason": candidate_layer_snapshot["best_verify_missing_reason"],
        "verify_lineage_total_count": len(verify_lineage_audit),
        "verify_lineage_generated_count": verify_lineage_generated_count,
        "verify_lineage_before_selector_count": verify_lineage_before_selector_count,
        "verify_lineage_entered_selector_count": verify_lineage_entered_selector_count,
        "verify_lineage_injected_before_selector_count": verify_lineage_injected_before_selector_count,
        "verify_lineage_audit": verify_lineage_audit,
        "remaining_scan_action_count": len(remaining_scan_actions),
        "remaining_verify_action_count": len(remaining_verify_actions),
        "remaining_scan_actions": remaining_scan_actions,
        "remaining_verify_actions": remaining_verify_actions,
        "final_scan_lineage_keys": _family_first_actions(final_candidates, "scan"),
        "final_verify_lineage_keys": _family_first_actions(final_candidates, "verify"),
        "selector_scan_lineage_keys": _family_first_actions(selector_candidates, "scan"),
        "selector_verify_lineage_keys": _family_first_actions(selector_candidates, "verify"),
        "template_scan_candidate_count": _count_candidates_by_family(template_candidates, "scan"),
        "template_verify_candidate_count": _count_candidates_by_family(template_candidates, "verify"),
        "llm_scan_candidate_count": _count_candidates_by_family(llm_candidates, "scan"),
        "llm_verify_candidate_count": _count_candidates_by_family(llm_candidates, "verify"),
        "final_scan_candidate_count": _count_candidates_by_family(final_candidates, "scan"),
        "final_verify_candidate_count": _count_candidates_by_family(final_candidates, "verify"),
        "selector_scan_candidate_count": _count_candidates_by_family(selector_candidates, "scan"),
        "selector_verify_candidate_count": _count_candidates_by_family(selector_candidates, "verify"),
        "oracle_tool_choice_applicable": oracle_tool_choice_applicable,
        "oracle_tool_optimal_family": oracle_tool_optimal_family,
        "oracle_best_scan_action": oracle_best_scan["action"] if oracle_best_scan else None,
        "oracle_best_scan_eval_score": _round_optional(
            oracle_best_scan["eval_score"] if oracle_best_scan else None
        ),
        "oracle_best_verify_action": oracle_best_verify["action"] if oracle_best_verify else None,
        "oracle_best_verify_eval_score": _round_optional(
            oracle_best_verify["eval_score"] if oracle_best_verify else None
        ),
        "oracle_tool_family_gap": _round_optional(
            abs(float(oracle_best_scan["eval_score"]) - float(oracle_best_verify["eval_score"]))
            if oracle_tool_choice_applicable and oracle_best_scan and oracle_best_verify
            else None
        ),
        "template_oracle_scan_action_covered": template_oracle_scan_action_covered,
        "template_oracle_verify_action_covered": template_oracle_verify_action_covered,
        "llm_oracle_scan_action_covered": llm_oracle_scan_action_covered,
        "llm_oracle_verify_action_covered": llm_oracle_verify_action_covered,
        "final_oracle_scan_action_covered": final_oracle_scan_action_covered,
        "final_oracle_verify_action_covered": final_oracle_verify_action_covered,
        "selector_oracle_scan_action_covered": selector_oracle_scan_action_covered,
        "selector_oracle_verify_action_covered": selector_oracle_verify_action_covered,
        "final_oracle_family_alignment": final_oracle_family_alignment,
        "selected_vs_oracle_family_mismatch": selected_vs_oracle_family_mismatch,
        "selected_vs_oracle_mismatch_kind": selected_vs_oracle_mismatch_kind,
        "tool_family_error_source": tool_family_error_source,
        "final_scan_eval_gap": _round_optional(final_scan_eval_gap),
        "final_verify_eval_gap": _round_optional(final_verify_eval_gap),
        "selector_family_applicable": selector_family_applicable,
        "selector_total_optimal_family": selector_total_optimal_family,
        "selector_family_override": selector_family_override,
        "regretful_ask": regretful_ask,
        "choose_regret": choose_regret,
        "over_ask_cost": _round(over_ask_cost),
        "under_ask_cost": _round(under_ask_cost),
        "selector_prefers_ask_nonpositive_advantage": selector_prefers_ask_nonpositive,
        "retrieved_memory_count": len(retrieved_memory_ids),
        "retrieved_choose_suppressing_memory": bool(choose_suppressing_memory_ids),
        "retrieved_ask_suppressing_memory": bool(ask_suppressing_memory_ids),
        "retrieved_choose_suppressing_memory_ids": choose_suppressing_memory_ids,
        "retrieved_ask_suppressing_memory_ids": ask_suppressing_memory_ids,
        "ask_gate_enabled": ask_gate_enabled,
        "ask_gate_allowed": bool(ask_gate_allowed) if ask_gate_enabled else None,
        "ask_gate_score": _round_optional(step_record.get("ask_gate_score")),
        "ask_gate_reason": ask_gate_reason or None,
        "pre_selector_filter_applied": bool(candidate_layer_snapshot["pre_selector_filter_applied"]),
        "pre_selector_filtered_pairs_empty": bool(
            candidate_layer_snapshot["pre_selector_filtered_pairs_empty"]
        ),
        "pre_selector_has_direct_evidence": bool(
            candidate_layer_snapshot["pre_selector_has_direct_evidence"]
        ),
        "pre_selector_original_candidate_count": int(
            step_record.get("pre_selector_original_candidate_count", len(final_candidates))
        ),
        "pre_selector_filtered_candidate_count": int(
            step_record.get("pre_selector_filtered_candidate_count", len(selector_candidates))
        ),
        "retrieved_positive_evidence": bool(positive_lessons),
        "retrieved_caution_evidence": bool(caution_lessons),
        "retrieved_helpful_ask": "ask_helpful" in retrieved_lesson_types,
        "retrieved_choose_regret": "choose_regretful" in retrieved_lesson_types,
        "retrieved_regretful_ask": "ask_regretful" in retrieved_lesson_types,
        "retrieved_choose_helpful": "choose_helpful" in retrieved_lesson_types,
        "written_decision_lesson_type": str(
            (step_record.get("written_decision_lesson") or {}).get("lesson_type", "")
        ),
        "high_margin": margin >= confidence_threshold,
        "margin": _round(margin),
        "confidence_threshold": _round(confidence_threshold),
        "selected_choose_without_prior_ask": bool(selected_type == "CHOOSE" and asked_count == 0),
        "choose_regret_without_prior_ask": bool(choose_regret and asked_count == 0),
    }


def _episode_rows_for_records(
    *,
    seed: int,
    mode_name: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    steps_by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("record_type") == "step":
            steps_by_task.setdefault(str(record.get("task_id")), []).append(record)

    episode_rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("record_type") != "episode":
            continue
        task_id = str(record.get("task_id"))
        task_steps = steps_by_task.get(task_id, [])
        ask_count = sum(
            1
            for step in task_steps
            if _action_type(str(step.get("action", ""))) == "ASK"
        )
        reflection_written = record.get("reflection_written") or {}
        reflection = reflection_written or record.get("reflection") or {}
        avoid_actions = [str(action) for action in reflection.get("avoid_actions", [])]
        z_delta = {
            field: float(reflection.get("z_delta", {}).get(field, 0.0))
            for field in REFLECTION_FIELDS
        }
        episode_rows.append(
            {
                "seed": seed,
                "mode": mode_name,
                "task_id": task_id,
                "task_index": _task_index(task_id),
                "ask_count": float(ask_count),
                "steps": float(record.get("steps", len(task_steps))),
                "reward": float(record.get("total_reward", 0.0)),
                "success": float(bool(record.get("success"))),
                "reflection_admitted": bool(record.get("reflection_admitted")),
                "reflection_backend": str(record.get("reflection_backend", "")),
                "reflection_written_present": bool(record.get("reflection_written")),
                "avoid_actions": avoid_actions,
                "avoid_choose_flag": float(any(_suppresses_choose(action) for action in avoid_actions)),
                "avoid_ask_flag": float(any(_suppresses_ask(action) for action in avoid_actions)),
                "z_delta": z_delta,
                "self_model_before_episode": record.get("self_model_before_episode"),
                "self_model_after": record.get("self_model_after"),
                "self_model_delta_applied": record.get("self_model_delta_applied"),
                "reflection_avoid_actions_written": record.get("reflection_avoid_actions_written"),
                "reflection_z_delta_written": record.get("reflection_z_delta_written"),
            }
        )
    episode_rows.sort(key=lambda row: (row["seed"], row["task_index"]))
    return episode_rows


def _summarize_step_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    applicable_rows = [row for row in rows if row["ask_advantage"] is not None]
    selected_ask_rows = [row for row in rows if row["selected_type"] == "ASK"]
    selected_choose_rows = [row for row in rows if row["selected_type"] == "CHOOSE"]
    selected_scan_rows = [row for row in rows if row["selected_tool_family"] == "scan"]
    selected_verify_rows = [row for row in rows if row["selected_tool_family"] == "verify"]
    selected_tool_rows = [
        row
        for row in rows
        if row["selected_tool_family"] in {"scan", "verify"}
    ]
    tool_choice_rows = [row for row in selected_tool_rows if row["tool_choice_applicable"]]
    wrong_tool_rows = [row for row in tool_choice_rows if row["wrong_tool_choice"]]
    applicable_selected_ask_rows = [row for row in applicable_rows if row["selected_type"] == "ASK"]
    applicable_selected_choose_rows = [row for row in applicable_rows if row["selected_type"] == "CHOOSE"]
    regretful_asks = [row for row in applicable_selected_ask_rows if row["regretful_ask"]]
    choose_regrets = [row for row in applicable_selected_choose_rows if row["choose_regret"]]
    nonpositive_advantage_rows = [row for row in applicable_rows if float(row["ask_advantage"]) <= 0]
    positive_advantage_rows = [row for row in applicable_rows if float(row["ask_advantage"]) > 0]
    selected_ask_nonpositive = [
        row
        for row in selected_ask_rows
        if row["ask_advantage"] is not None and float(row["ask_advantage"]) <= 0
    ]
    selector_overrides = [
        row
        for row in selected_ask_nonpositive
        if row["selector_prefers_ask_nonpositive_advantage"]
    ]
    retrieved_memory_selected_ask = [
        row
        for row in selected_ask_rows
        if int(row["retrieved_memory_count"]) > 0
    ]
    regretful_over_ask_costs = [float(row["over_ask_cost"]) for row in regretful_asks]
    under_ask_costs = [float(row["under_ask_cost"]) for row in choose_regrets]
    over_tool_costs = [float(row["over_tool_cost"]) for row in wrong_tool_rows]
    under_tool_costs = [float(row["under_tool_cost"]) for row in wrong_tool_rows]
    ask_advantages = [float(row["ask_advantage"]) for row in applicable_rows]
    gate_enabled_rows = [row for row in rows if row["ask_gate_enabled"]]
    gate_decision_rows = [
        row
        for row in gate_enabled_rows
        if row.get("ask_gate_reason") not in {None, "bypass_missing_ask_or_choose"}
    ]
    gate_blocked_rows = [row for row in gate_decision_rows if not bool(row["ask_gate_allowed"])]
    gate_positive_rows = [
        row
        for row in gate_decision_rows
        if row["ask_advantage"] is not None and float(row["ask_advantage"]) > 0
    ]
    gate_nonpositive_rows = [
        row
        for row in gate_decision_rows
        if row["ask_advantage"] is not None and float(row["ask_advantage"]) <= 0
    ]
    false_blocks = [
        row
        for row in gate_blocked_rows
        if row["ask_advantage"] is not None and float(row["ask_advantage"]) > 0
    ]
    false_allows = [
        row
        for row in selected_ask_rows
        if row["ask_gate_enabled"]
        and row["ask_advantage"] is not None
        and float(row["ask_advantage"]) <= 0
    ]
    template_backend_rows = [
        row for row in rows if str(row.get("candidate_backend", "")).startswith("template")
    ]
    openai_backend_rows = [
        row
        for row in rows
        if str(row.get("candidate_backend", "")) in {"openai_responses", "openai_responses_plus_template"}
    ]
    union_backend_rows = [
        row for row in rows if str(row.get("candidate_backend", "")) == "openai_responses_plus_template"
    ]
    scan_available_rows = [row for row in rows if int(row["remaining_scan_action_count"]) > 0]
    verify_available_rows = [row for row in rows if int(row["remaining_verify_action_count"]) > 0]
    oracle_tool_rows = [row for row in rows if row["oracle_tool_choice_applicable"]]
    selector_family_rows = [row for row in rows if row["selector_family_applicable"]]
    selector_family_override_rows = [row for row in selector_family_rows if row["selector_family_override"]]
    final_oracle_mismatch_rows = [
        row
        for row in oracle_tool_rows
        if not bool(row["final_oracle_family_alignment"])
    ]
    selected_vs_oracle_mismatch_rows = [
        row
        for row in oracle_tool_rows
        if bool(row["selected_vs_oracle_family_mismatch"])
    ]
    choose_swallow_rows = [
        row
        for row in selected_vs_oracle_mismatch_rows
        if row["selected_vs_oracle_mismatch_kind"] == "choose_swallow"
    ]
    tool_family_mismatch_vs_oracle_rows = [
        row
        for row in selected_vs_oracle_mismatch_rows
        if row["selected_vs_oracle_mismatch_kind"] == "tool_family_mismatch"
    ]
    surface_family_distortion_rows = [
        row
        for row in oracle_tool_rows
        if row["tool_family_error_source"] == "surface_family_distortion"
    ]
    policy_family_error_rows = [
        row
        for row in oracle_tool_rows
        if row["tool_family_error_source"] == "policy_family_error"
    ]
    mixed_surface_and_policy_error_rows = [
        row
        for row in oracle_tool_rows
        if row["tool_family_error_source"] == "mixed_surface_and_policy_error"
    ]
    best_scan_surface_rows = [
        row for row in oracle_tool_rows if row["best_scan_surface_shape"] != "none"
    ]
    best_verify_surface_rows = [
        row for row in oracle_tool_rows if row["best_verify_surface_shape"] != "none"
    ]
    best_scan_surface_rows_before = [
        row for row in scan_available_rows if row["best_scan_surface_shape_before"] != "none"
    ]
    best_verify_surface_rows_before = [
        row for row in verify_available_rows if row["best_verify_surface_shape_before"] != "none"
    ]
    best_scan_surface_rows_after = [
        row for row in scan_available_rows if row["best_scan_surface_shape_after"] != "none"
    ]
    best_verify_surface_rows_after = [
        row for row in verify_available_rows if row["best_verify_surface_shape_after"] != "none"
    ]
    verify_exposed_rows = [
        row for row in verify_available_rows if bool(row["verify_candidate_exposed"])
    ]
    verify_missing_reason_not_generated_rows = [
        row for row in verify_available_rows if row["best_verify_missing_reason"] == "not_generated"
    ]
    verify_missing_reason_generated_but_pruned_rows = [
        row for row in verify_available_rows if row["best_verify_missing_reason"] == "generated_but_pruned"
    ]
    verify_missing_reason_generated_noncanonical_rows = [
        row for row in verify_available_rows if row["best_verify_missing_reason"] == "generated_noncanonical"
    ]
    verify_missing_reason_dominated_after_scoring_rows = [
        row for row in verify_available_rows if row["best_verify_missing_reason"] == "dominated_after_scoring"
    ]
    oracle_family_gaps = [float(row["oracle_tool_family_gap"]) for row in oracle_tool_rows]
    final_scan_eval_gaps = [
        float(row["final_scan_eval_gap"])
        for row in rows
        if row["final_scan_eval_gap"] is not None
    ]
    final_verify_eval_gaps = [
        float(row["final_verify_eval_gap"])
        for row in rows
        if row["final_verify_eval_gap"] is not None
    ]
    selector_family_margins = [
        abs(float(row["best_scan_total_score"]) - float(row["best_verify_total_score"]))
        for row in selector_family_rows
        if row["best_scan_total_score"] is not None and row["best_verify_total_score"] is not None
    ]

    return {
        "step_count": len(rows),
        "applicable_step_count": len(applicable_rows),
        "selected_ask_count": len(selected_ask_rows),
        "selected_choose_count": len(selected_choose_rows),
        "selected_scan_count": len(selected_scan_rows),
        "selected_verify_count": len(selected_verify_rows),
        "selected_tool_count": len(selected_tool_rows),
        "selected_ask_rate": _round(_safe_div(len(selected_ask_rows), len(rows))),
        "selected_choose_rate": _round(_safe_div(len(selected_choose_rows), len(rows))),
        "candidate_backend_template_rate": _round(_safe_div(len(template_backend_rows), len(rows))),
        "candidate_backend_openai_rate": _round(_safe_div(len(openai_backend_rows), len(rows))),
        "candidate_backend_union_rate": _round(_safe_div(len(union_backend_rows), len(rows))),
        "remaining_scan_action_mean": _round(_mean([float(row["remaining_scan_action_count"]) for row in rows])),
        "remaining_verify_action_mean": _round(_mean([float(row["remaining_verify_action_count"]) for row in rows])),
        "before_normalization_scan_candidate_mean": _round(
            _mean([float(row["before_normalization_scan_candidate_count"]) for row in rows])
        ),
        "before_normalization_verify_candidate_mean": _round(
            _mean([float(row["before_normalization_verify_candidate_count"]) for row in rows])
        ),
        "after_normalization_scan_candidate_mean": _round(
            _mean([float(row["after_normalization_scan_candidate_count"]) for row in rows])
        ),
        "after_normalization_verify_candidate_mean": _round(
            _mean([float(row["after_normalization_verify_candidate_count"]) for row in rows])
        ),
        "scan_usage_rate": _round(_safe_div(len(selected_scan_rows), len(rows))),
        "verify_usage_rate": _round(_safe_div(len(selected_verify_rows), len(rows))),
        "scan_share_given_tool_use": _round(_safe_div(len(selected_scan_rows), len(selected_tool_rows))),
        "verify_share_given_tool_use": _round(_safe_div(len(selected_verify_rows), len(selected_tool_rows))),
        "template_scan_family_coverage_rate": _round(
            _safe_div(
                sum(int(int(row["template_scan_candidate_count"]) > 0) for row in scan_available_rows),
                len(scan_available_rows),
            )
        ),
        "template_verify_family_coverage_rate": _round(
            _safe_div(
                sum(int(int(row["template_verify_candidate_count"]) > 0) for row in verify_available_rows),
                len(verify_available_rows),
            )
        ),
        "llm_scan_family_coverage_rate": _round(
            _safe_div(
                sum(int(int(row["llm_scan_candidate_count"]) > 0) for row in scan_available_rows),
                len(scan_available_rows),
            )
        ),
        "llm_verify_family_coverage_rate": _round(
            _safe_div(
                sum(int(int(row["llm_verify_candidate_count"]) > 0) for row in verify_available_rows),
                len(verify_available_rows),
            )
        ),
        "final_scan_family_coverage_rate": _round(
            _safe_div(
                sum(int(int(row["final_scan_candidate_count"]) > 0) for row in scan_available_rows),
                len(scan_available_rows),
            )
        ),
        "final_verify_family_coverage_rate": _round(
            _safe_div(
                sum(int(int(row["final_verify_candidate_count"]) > 0) for row in verify_available_rows),
                len(verify_available_rows),
            )
        ),
        "oracle_tool_choice_step_count": len(oracle_tool_rows),
        "oracle_family_gap_mean": _round(_mean(oracle_family_gaps)),
        "template_oracle_scan_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["template_oracle_scan_action_covered"])) for row in scan_available_rows),
                len(scan_available_rows),
            )
        ),
        "template_oracle_verify_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["template_oracle_verify_action_covered"])) for row in verify_available_rows),
                len(verify_available_rows),
            )
        ),
        "llm_oracle_scan_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["llm_oracle_scan_action_covered"])) for row in scan_available_rows),
                len(scan_available_rows),
            )
        ),
        "llm_oracle_verify_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["llm_oracle_verify_action_covered"])) for row in verify_available_rows),
                len(verify_available_rows),
            )
        ),
        "final_oracle_scan_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["final_oracle_scan_action_covered"])) for row in scan_available_rows),
                len(scan_available_rows),
            )
        ),
        "final_oracle_verify_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["final_oracle_verify_action_covered"])) for row in verify_available_rows),
                len(verify_available_rows),
            )
        ),
        "selector_oracle_scan_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["selector_oracle_scan_action_covered"])) for row in scan_available_rows),
                len(scan_available_rows),
            )
        ),
        "selector_oracle_verify_coverage_rate": _round(
            _safe_div(
                sum(int(bool(row["selector_oracle_verify_action_covered"])) for row in verify_available_rows),
                len(verify_available_rows),
            )
        ),
        "final_oracle_family_alignment_rate": _round(
            _safe_div(
                sum(int(bool(row["final_oracle_family_alignment"])) for row in oracle_tool_rows),
                len(oracle_tool_rows),
            )
        ),
        "final_oracle_family_mismatch_rate": _round(
            _safe_div(len(final_oracle_mismatch_rows), len(oracle_tool_rows))
        ),
        "selected_vs_oracle_mismatch_count": len(selected_vs_oracle_mismatch_rows),
        "selected_vs_oracle_family_mismatch_rate": _round(
            _safe_div(len(selected_vs_oracle_mismatch_rows), len(oracle_tool_rows))
        ),
        "choose_swallow_count": len(choose_swallow_rows),
        "choose_swallow_rate_given_oracle_tool_choice": _round(
            _safe_div(len(choose_swallow_rows), len(oracle_tool_rows))
        ),
        "tool_family_mismatch_rate_vs_oracle": _round(
            _safe_div(len(tool_family_mismatch_vs_oracle_rows), len(oracle_tool_rows))
        ),
        "surface_family_distortion_count": len(surface_family_distortion_rows),
        "surface_family_distortion_rate": _round(
            _safe_div(len(surface_family_distortion_rows), len(oracle_tool_rows))
        ),
        "policy_family_error_count": len(policy_family_error_rows),
        "policy_family_error_rate": _round(
            _safe_div(len(policy_family_error_rows), len(oracle_tool_rows))
        ),
        "mixed_surface_and_policy_error_count": len(mixed_surface_and_policy_error_rows),
        "mixed_surface_and_policy_error_rate": _round(
            _safe_div(len(mixed_surface_and_policy_error_rows), len(oracle_tool_rows))
        ),
        "candidate_related_share": _round(
            _safe_div(
                len(choose_swallow_rows)
                + len(surface_family_distortion_rows)
                + len(mixed_surface_and_policy_error_rows),
                len(selected_vs_oracle_mismatch_rows),
            )
        ),
        "policy_family_error_share": _round(
            _safe_div(len(policy_family_error_rows), len(selected_vs_oracle_mismatch_rows))
        ),
        "best_scan_canonical_surface_rate_before": _round(
            _safe_div(
                sum(int(row["best_scan_surface_shape_before"] == "canonical_choose_best") for row in best_scan_surface_rows_before),
                len(best_scan_surface_rows_before),
            )
        ),
        "best_verify_canonical_surface_rate_before": _round(
            _safe_div(
                sum(int(row["best_verify_surface_shape_before"] == "canonical_choose_best") for row in best_verify_surface_rows_before),
                len(best_verify_surface_rows_before),
            )
        ),
        "best_scan_canonical_surface_rate": _round(
            _safe_div(
                sum(int(row["best_scan_surface_shape"] == "canonical_choose_best") for row in best_scan_surface_rows),
                len(best_scan_surface_rows),
            )
        ),
        "best_scan_fixed_surface_rate": _round(
            _safe_div(
                sum(int(row["best_scan_surface_shape"] == "fixed_choose") for row in best_scan_surface_rows),
                len(best_scan_surface_rows),
            )
        ),
        "best_scan_ask_only_surface_rate": _round(
            _safe_div(
                sum(int(row["best_scan_surface_shape"] == "ask_only") for row in best_scan_surface_rows),
                len(best_scan_surface_rows),
            )
        ),
        "best_scan_multi_step_surface_rate": _round(
            _safe_div(
                sum(int(row["best_scan_surface_shape"] == "multi_step") for row in best_scan_surface_rows),
                len(best_scan_surface_rows),
            )
        ),
        "best_verify_canonical_surface_rate": _round(
            _safe_div(
                sum(int(row["best_verify_surface_shape"] == "canonical_choose_best") for row in best_verify_surface_rows),
                len(best_verify_surface_rows),
            )
        ),
        "best_verify_fixed_surface_rate": _round(
            _safe_div(
                sum(int(row["best_verify_surface_shape"] == "fixed_choose") for row in best_verify_surface_rows),
                len(best_verify_surface_rows),
            )
        ),
        "best_verify_ask_only_surface_rate": _round(
            _safe_div(
                sum(int(row["best_verify_surface_shape"] == "ask_only") for row in best_verify_surface_rows),
                len(best_verify_surface_rows),
            )
        ),
        "best_verify_multi_step_surface_rate": _round(
            _safe_div(
                sum(int(row["best_verify_surface_shape"] == "multi_step") for row in best_verify_surface_rows),
                len(best_verify_surface_rows),
            )
        ),
        "best_scan_canonical_surface_rate_after": _round(
            _safe_div(
                sum(int(row["best_scan_surface_shape_after"] == "canonical_choose_best") for row in best_scan_surface_rows_after),
                len(best_scan_surface_rows_after),
            )
        ),
        "best_verify_canonical_surface_rate_after": _round(
            _safe_div(
                sum(int(row["best_verify_surface_shape_after"] == "canonical_choose_best") for row in best_verify_surface_rows_after),
                len(best_verify_surface_rows_after),
            )
        ),
        "verify_candidate_exposed_rate": _round(
            _safe_div(len(verify_exposed_rows), len(verify_available_rows))
        ),
        "best_verify_missing_reason_not_generated_rate": _round(
            _safe_div(len(verify_missing_reason_not_generated_rows), len(verify_available_rows))
        ),
        "best_verify_missing_reason_generated_but_pruned_rate": _round(
            _safe_div(len(verify_missing_reason_generated_but_pruned_rows), len(verify_available_rows))
        ),
        "best_verify_missing_reason_generated_noncanonical_rate": _round(
            _safe_div(len(verify_missing_reason_generated_noncanonical_rows), len(verify_available_rows))
        ),
        "best_verify_missing_reason_dominated_after_scoring_rate": _round(
            _safe_div(len(verify_missing_reason_dominated_after_scoring_rows), len(verify_available_rows))
        ),
        "final_scan_eval_gap_mean": _round(_mean(final_scan_eval_gaps)),
        "final_verify_eval_gap_mean": _round(_mean(final_verify_eval_gaps)),
        "selector_family_step_count": len(selector_family_rows),
        "selector_family_override_rate": _round(
            _safe_div(len(selector_family_override_rows), len(selector_family_rows))
        ),
        "selector_family_margin_mean": _round(_mean(selector_family_margins)),
        "selected_choose_without_prior_ask_rate": _round(
            _safe_div(
                sum(int(row["selected_choose_without_prior_ask"]) for row in selected_choose_rows),
                len(selected_choose_rows),
            )
        ),
        "regretful_ask_count": len(regretful_asks),
        "regretful_ask_rate": _round(_safe_div(len(regretful_asks), len(applicable_selected_ask_rows))),
        "choose_regret_count": len(choose_regrets),
        "choose_regret_rate": _round(_safe_div(len(choose_regrets), len(applicable_selected_choose_rows))),
        "choose_regret_without_prior_ask_rate": _round(
            _safe_div(
                sum(int(row["choose_regret_without_prior_ask"]) for row in choose_regrets),
                len(choose_regrets),
            )
        ),
        "ask_advantage_mean": _round(_mean(ask_advantages)),
        "ask_advantage_median": _round(_median(ask_advantages)),
        "nonpositive_ask_advantage_rate": _round(_safe_div(len(nonpositive_advantage_rows), len(applicable_rows))),
        "positive_ask_advantage_rate": _round(_safe_div(len(positive_advantage_rows), len(applicable_rows))),
        "over_ask_cost_total": _round(sum(regretful_over_ask_costs)),
        "over_ask_cost_mean_per_regretful_ask": _round(_mean(regretful_over_ask_costs)),
        "over_ask_cost_mean_per_step": _round(_safe_div(sum(regretful_over_ask_costs), len(rows))),
        "under_ask_cost_total": _round(sum(under_ask_costs)),
        "under_ask_cost_mean_per_choose_regret": _round(_mean(under_ask_costs)),
        "under_ask_cost_mean_per_step": _round(_safe_div(sum(under_ask_costs), len(rows))),
        "tool_choice_step_count": len(tool_choice_rows),
        "tool_precision": _round(
            _safe_div(sum(int(row["correct_tool_choice"]) for row in tool_choice_rows), len(tool_choice_rows))
        ),
        "wrong_tool_count": len(wrong_tool_rows),
        "wrong_tool_rate": _round(_safe_div(len(wrong_tool_rows), len(tool_choice_rows))),
        "over_tool_cost_total": _round(sum(over_tool_costs)),
        "over_tool_cost_mean_per_wrong_tool": _round(_mean(over_tool_costs)),
        "over_tool_cost_mean_per_step": _round(_safe_div(sum(over_tool_costs), len(rows))),
        "under_tool_cost_total": _round(sum(under_tool_costs)),
        "under_tool_cost_mean_per_wrong_tool": _round(_mean(under_tool_costs)),
        "under_tool_cost_mean_per_step": _round(_safe_div(sum(under_tool_costs), len(rows))),
        "selector_override_count": len(selector_overrides),
        "selector_override_rate_given_nonpositive_ask_advantage": _round(
            _safe_div(len(selector_overrides), len(selected_ask_nonpositive))
        ),
        "selected_ask_with_choose_suppressing_memory_rate": _round(
            _safe_div(
                sum(int(row["retrieved_choose_suppressing_memory"]) for row in retrieved_memory_selected_ask),
                len(retrieved_memory_selected_ask),
            )
        ),
        "regretful_ask_with_choose_suppressing_memory_rate": _round(
            _safe_div(
                sum(int(row["retrieved_choose_suppressing_memory"]) for row in regretful_asks),
                len(regretful_asks),
            )
        ),
        "regretful_ask_high_margin_share": _round(
            _safe_div(sum(int(row["high_margin"]) for row in regretful_asks), len(regretful_asks))
        ),
        "regretful_ask_easy_share": _round(
            _safe_div(sum(int(row["difficulty"] == "easy") for row in regretful_asks), len(regretful_asks))
        ),
        "regretful_ask_low_noise_share": _round(
            _safe_div(sum(int(row["noise_level"] in {"0.00", "0.05", "0.10"}) for row in regretful_asks), len(regretful_asks))
        ),
        "choose_regret_high_margin_share": _round(
            _safe_div(sum(int(row["high_margin"]) for row in choose_regrets), len(choose_regrets))
        ),
        "choose_regret_without_prior_ask_share": _round(
            _safe_div(sum(int(row["choose_regret_without_prior_ask"]) for row in choose_regrets), len(choose_regrets))
        ),
        "ask_gate_step_count": len(gate_enabled_rows),
        "ask_gate_decision_count": len(gate_decision_rows),
        "ask_gate_block_rate": _round(_safe_div(len(gate_blocked_rows), len(gate_decision_rows))),
        "ask_gate_allow_rate_given_positive_ask_advantage": _round(
            _safe_div(
                sum(int(bool(row["ask_gate_allowed"])) for row in gate_positive_rows),
                len(gate_positive_rows),
            )
        ),
        "ask_gate_block_rate_given_nonpositive_ask_advantage": _round(
            _safe_div(
                sum(int(not bool(row["ask_gate_allowed"])) for row in gate_nonpositive_rows),
                len(gate_nonpositive_rows),
            )
        ),
        "false_block_rate": _round(_safe_div(len(false_blocks), len(gate_blocked_rows))),
        "false_allow_rate": _round(_safe_div(len(false_allows), len(selected_ask_rows))),
        "retrieved_positive_evidence_rate": _round(
            _safe_div(sum(int(row["retrieved_positive_evidence"]) for row in rows), len(rows))
        ),
        "retrieved_caution_evidence_rate": _round(
            _safe_div(sum(int(row["retrieved_caution_evidence"]) for row in rows), len(rows))
        ),
        "retrieved_helpful_ask_rate": _round(
            _safe_div(sum(int(row["retrieved_helpful_ask"]) for row in rows), len(rows))
        ),
        "retrieved_choose_regret_rate": _round(
            _safe_div(sum(int(row["retrieved_choose_regret"]) for row in rows), len(rows))
        ),
        "retrieved_regretful_ask_rate": _round(
            _safe_div(sum(int(row["retrieved_regretful_ask"]) for row in rows), len(rows))
        ),
        "retrieved_choose_helpful_rate": _round(
            _safe_div(sum(int(row["retrieved_choose_helpful"]) for row in rows), len(rows))
        ),
    }


def _summarize_verify_survival_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lineage_count = len(rows)
    summary = {
        "verify_lineage_count": lineage_count,
        "verify_generated_count": sum(int(row["verify_generated"]) for row in rows),
        "verify_normalized_count": sum(int(row["verify_normalized"]) for row in rows),
        "verify_retained_after_family_merge_count": sum(
            int(row["verify_retained_after_family_merge"]) for row in rows
        ),
        "verify_retained_after_candidate_cap_count": sum(
            int(row["verify_retained_after_candidate_cap"]) for row in rows
        ),
        "verify_retained_after_dedup_count": sum(
            int(row["verify_retained_after_dedup"]) for row in rows
        ),
        "verify_retained_before_selector_count": sum(
            int(row["verify_retained_before_selector"]) for row in rows
        ),
        "verify_entered_selector_count": sum(int(row["verify_entered_selector"]) for row in rows),
        "verify_injected_before_selector_count": sum(
            int(row.get("verify_injected_before_selector", False)) for row in rows
        ),
        "verify_shape_replacement_count": sum(
            int(row.get("verify_shape_replacement_detected", False)) for row in rows
        ),
    }
    summary.update(
        {
            "verify_generated_rate": _round(_safe_div(summary["verify_generated_count"], lineage_count)),
            "verify_normalized_rate": _round(_safe_div(summary["verify_normalized_count"], lineage_count)),
            "verify_retained_after_family_merge_rate": _round(
                _safe_div(summary["verify_retained_after_family_merge_count"], lineage_count)
            ),
            "verify_retained_after_candidate_cap_rate": _round(
                _safe_div(summary["verify_retained_after_candidate_cap_count"], lineage_count)
            ),
            "verify_retained_after_dedup_rate": _round(
                _safe_div(summary["verify_retained_after_dedup_count"], lineage_count)
            ),
            "verify_retained_before_selector_rate": _round(
                _safe_div(summary["verify_retained_before_selector_count"], lineage_count)
            ),
            "verify_entered_selector_rate": _round(
                _safe_div(summary["verify_entered_selector_count"], lineage_count)
            ),
            "verify_injected_before_selector_rate": _round(
                _safe_div(summary["verify_injected_before_selector_count"], lineage_count)
            ),
            "verify_shape_replacement_rate": _round(
                _safe_div(summary["verify_shape_replacement_count"], lineage_count)
            ),
        }
    )
    for stage in VERIFY_PRUNED_STAGES:
        count = sum(int(str(row["verify_pruned_stage"]) == stage) for row in rows)
        summary[f"verify_pruned_stage_{stage}_count"] = count
        summary[f"verify_pruned_stage_{stage}_rate"] = _round(_safe_div(count, lineage_count))
    for reason in VERIFY_PRUNED_REASONS:
        count = sum(int(str(row["verify_pruned_reason"]) == reason) for row in rows)
        summary[f"verify_pruned_reason_{reason}_count"] = count
        summary[f"verify_pruned_reason_{reason}_rate"] = _round(_safe_div(count, lineage_count))
    stage_specs = (
        ("generated", lambda row: True, lambda row: bool(row["verify_generated"])),
        (
            "family_merge",
            lambda row: bool(row["verify_generated"]),
            lambda row: bool(row["verify_generated"]) and bool(row["verify_retained_after_family_merge"]),
        ),
        (
            "normalization_merge",
            lambda row: bool(row["verify_retained_after_family_merge"]),
            lambda row: bool(row["verify_retained_after_family_merge"]) and bool(row["verify_normalized"]),
        ),
        (
            "dedup",
            lambda row: bool(row["verify_normalized"]),
            lambda row: bool(row["verify_normalized"]) and bool(row["verify_retained_after_dedup"]),
        ),
        (
            "candidate_cap",
            lambda row: True,
            lambda row: bool(row["verify_retained_after_candidate_cap"]),
        ),
        (
            "pre_selector_filter",
            lambda row: bool(row["verify_retained_before_selector"]),
            lambda row: bool(row["verify_retained_before_selector"]) and bool(row["verify_entered_selector"]),
        ),
        (
            "selector_entry",
            lambda row: bool(row["verify_retained_before_selector"]),
            lambda row: bool(row["verify_entered_selector"]),
        ),
    )
    for stage_name, applicable_fn, retained_fn in stage_specs:
        applicable_count = sum(int(applicable_fn(row)) for row in rows)
        retained_count = sum(int(retained_fn(row)) for row in rows)
        summary[f"{stage_name}_denominator"] = lineage_count
        summary[f"{stage_name}_applicable_count"] = applicable_count
        summary[f"{stage_name}_retained_count"] = retained_count
        summary[f"{stage_name}_retained_rate"] = _round(_safe_div(retained_count, applicable_count))
    return summary


def _mean_optional_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return _round(_mean(values))


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _family_pre_selector_survival(step_rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    applicable_count = 0
    retained_count = 0
    for row in step_rows:
        remaining_actions = list(row.get(f"remaining_{family}_actions", []))
        final_keys = set(row.get(f"final_{family}_lineage_keys", []))
        selector_keys = set(row.get(f"selector_{family}_lineage_keys", []))
        seen: set[str] = set()
        for action in remaining_actions:
            action = str(action)
            if action in seen:
                continue
            seen.add(action)
            if action not in final_keys:
                continue
            applicable_count += 1
            if action in selector_keys:
                retained_count += 1
    return {
        "pre_selector_applicable_count": applicable_count,
        "pre_selector_retained_count": retained_count,
        "pre_selector_retained_rate": _round(_safe_div(retained_count, applicable_count)),
    }


def _collect_pre_selector_pruning_rows(
    step_rows_by_mode: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode_name, step_rows in step_rows_by_mode.items():
        flattened: list[dict[str, Any]] = []
        for row in step_rows:
            for lineage_row in row.get("verify_lineage_audit", []):
                if not bool(lineage_row.get("verify_retained_before_selector")):
                    continue
                blocked = not bool(lineage_row.get("verify_entered_selector"))
                if blocked:
                    block_source, block_reason = _classify_pre_selector_block(
                        ask_gate_enabled=bool(row.get("ask_gate_enabled")),
                        ask_gate_allowed=row.get("ask_gate_allowed"),
                        ask_gate_reason=row.get("ask_gate_reason"),
                        pre_selector_has_direct_evidence=bool(row.get("pre_selector_has_direct_evidence")),
                        pre_selector_filtered_pairs_empty=bool(row.get("pre_selector_filtered_pairs_empty")),
                    )
                else:
                    block_source, block_reason = None, None
                flattened.append(
                    {
                        "seed": row["seed"],
                        "mode": mode_name,
                        "task_id": row["task_id"],
                        "task_index": row["task_index"],
                        "step_index": row["step_index"],
                        "difficulty": row["difficulty"],
                        "noise_level": row["noise_level"],
                        "tool_profile": row["tool_profile"],
                        "tool_subtype": row["tool_subtype"],
                        "ask_family": "verify",
                        "lineage_key": lineage_row.get("verify_lineage_key"),
                        "verify_pruned_stage": lineage_row.get("verify_pruned_stage"),
                        "verify_pruned_reason": lineage_row.get("verify_pruned_reason"),
                        "verify_retained_before_selector": bool(
                            lineage_row.get("verify_retained_before_selector")
                        ),
                        "verify_entered_selector": bool(lineage_row.get("verify_entered_selector")),
                        "ask_gate_enabled": bool(row.get("ask_gate_enabled")),
                        "ask_gate_allowed": row.get("ask_gate_allowed"),
                        "ask_gate_reason": row.get("ask_gate_reason"),
                        "pre_selector_block_source": block_source,
                        "pre_selector_block_reason": block_reason,
                        "pre_selector_has_direct_evidence": bool(
                            row.get("pre_selector_has_direct_evidence")
                        ),
                        "pre_selector_filter_applied": bool(row.get("pre_selector_filter_applied")),
                        "pre_selector_filtered_pairs_empty": bool(
                            row.get("pre_selector_filtered_pairs_empty")
                        ),
                        "ask_gate_score": row.get("ask_gate_score"),
                        "ask_advantage": row.get("ask_advantage"),
                        "best_ask_family": row.get("ask_gate_best_ask_family"),
                        "best_ask_eval": row.get("ask_gate_best_ask_eval"),
                        "best_choose_eval": row.get("ask_gate_best_choose_eval"),
                        "voi_ratio": row.get("ask_gate_voi_ratio"),
                        "uncertainty": row.get("ask_gate_uncertainty"),
                        "base_uncertainty": row.get("ask_gate_base_uncertainty"),
                        "reliability_uncertainty": row.get("ask_gate_reliability_uncertainty"),
                        "evidence_conflict": row.get("ask_gate_evidence_conflict"),
                        "memory_signal": row.get("ask_gate_memory_signal"),
                        "helpful_ask": row.get("ask_gate_helpful_ask"),
                        "choose_regret": row.get("ask_gate_choose_regret"),
                        "regretful_ask": row.get("ask_gate_regretful_ask"),
                        "choose_helpful": row.get("ask_gate_choose_helpful"),
                    }
                )
        rows_by_mode[mode_name] = flattened
    return rows_by_mode


def _summarize_pre_selector_pruning(
    *,
    step_rows: list[dict[str, Any]],
    verify_survival_summary: dict[str, Any],
    pre_selector_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    filter_rule_rows = [
        row
        for row in pre_selector_rows
        if row.get("verify_pruned_reason") == "filter_rule"
    ]
    entered_selector_rows = [
        row
        for row in pre_selector_rows
        if bool(row.get("verify_entered_selector"))
    ]
    filter_rule_without_direct_gate_evidence = sum(
        int(not bool(row.get("pre_selector_has_direct_evidence")))
        for row in filter_rule_rows
    )
    secondary_comparison = {}
    for key in PRE_SELECTOR_SECONDARY_CONTEXT_FIELDS:
        secondary_comparison[key] = {
            "filter_rule_mean": _mean_optional_field(filter_rule_rows, key),
            "entered_selector_mean": _mean_optional_field(entered_selector_rows, key),
        }

    return {
        "claim_boundary": PRE_SELECTOR_CLAIM_BOUNDARY,
        "primary_evidence_fields": list(PRE_SELECTOR_PRIMARY_EVIDENCE_FIELDS),
        "secondary_context_fields": list(PRE_SELECTOR_SECONDARY_CONTEXT_FIELDS),
        "pre_selector_denominator": int(verify_survival_summary.get("pre_selector_filter_denominator", 0)),
        "pre_selector_applicable_count": int(
            verify_survival_summary.get("pre_selector_filter_applicable_count", 0)
        ),
        "pre_selector_retained_count": int(
            verify_survival_summary.get("pre_selector_filter_retained_count", 0)
        ),
        "pre_selector_retained_rate": _round(
            verify_survival_summary.get("pre_selector_filter_retained_rate", 0.0)
        ),
        "filter_rule_count": len(filter_rule_rows),
        "filter_rule_without_direct_gate_evidence": filter_rule_without_direct_gate_evidence,
        "filter_rule_ask_gate_reason_counts": _count_by_key(filter_rule_rows, "ask_gate_reason"),
        "filter_rule_tool_subtype_counts": _count_by_key(filter_rule_rows, "tool_subtype"),
        "secondary_context_comparison": secondary_comparison,
        "verify_vs_scan_pre_selector_survival": {
            "verify": _family_pre_selector_survival(step_rows, "verify"),
            "scan": _family_pre_selector_survival(step_rows, "scan"),
        },
    }


def _collect_blocked_step_audit_rows(
    step_contexts_by_mode: dict[str, list[dict[str, Any]]],
    step_rows_by_mode: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode_name, contexts in step_contexts_by_mode.items():
        flattened: list[dict[str, Any]] = []
        row_lookup = {
            (int(row["seed"]), str(row["task_id"]), int(row["step_index"])): row
            for row in step_rows_by_mode.get(mode_name, [])
        }
        for context in contexts:
            step_record = dict(context.get("step_record") or {})
            if str(step_record.get("ask_gate_reason")) != "strong_negative_voi":
                continue
            seed = int(context["seed"])
            task_id = str(step_record.get("task_id"))
            step_index = int(step_record.get("step", 0) or 0)
            step_row = row_lookup.get((seed, task_id, step_index))
            if step_row is None:
                continue
            task_config = context.get("task_config")
            replay = (
                _replay_env_to_step(
                    task_config=task_config,
                    prior_actions=list(context.get("prior_actions") or []),
                )
                if task_config is not None
                else None
            )
            env = replay[0] if replay is not None else None
            observation = replay[1] if replay is not None else {}
            remaining_ask_actions = [
                str(action)
                for action in observation.get("options", [])
                if str(action).startswith("ASK_SCAN_") or str(action).startswith("ASK_VERIFY_")
            ]
            oracle_best_ask = (
                _best_oracle_candidate_for_actions(env, actions=remaining_ask_actions)
                if env is not None
                else None
            )
            best_choose = _best_surface_eval_for_action_type(step_record, action_type="CHOOSE")
            best_surfaced_ask = _best_surface_eval_for_action_type(step_record, action_type="ASK")
            best_choose_eval = (
                float(best_choose["eval_score"]) if best_choose is not None else None
            )
            best_surfaced_ask_eval = (
                float(best_surfaced_ask["eval_score"]) if best_surfaced_ask is not None else None
            )
            oracle_best_ask_eval = (
                float(oracle_best_ask["eval_score"]) if oracle_best_ask is not None else None
            )
            blocked_step_class = _classify_blocked_step(
                oracle_best_ask_eval=oracle_best_ask_eval,
                best_choose_eval=best_choose_eval,
                best_surfaced_ask_eval=best_surfaced_ask_eval,
            )
            flattened.append(
                {
                    "seed": seed,
                    "mode": mode_name,
                    "task_id": task_id,
                    "task_index": int(step_row["task_index"]),
                    "step_index": step_index,
                    "difficulty": step_row.get("difficulty"),
                    "noise_level": step_row.get("noise_level"),
                    "tool_profile": step_row.get("tool_profile"),
                    "tool_subtype": step_row.get("tool_subtype"),
                    "selected_action": str(step_record.get("action", "")),
                    "selected_type": str(step_row.get("selected_type", "")),
                    "ask_gate_reason": "strong_negative_voi",
                    "best_choose_eval": _round_optional(best_choose_eval),
                    "best_surfaced_ask_eval": _round_optional(best_surfaced_ask_eval),
                    "oracle_best_ask_eval": _round_optional(oracle_best_ask_eval),
                    "oracle_best_ask_action": None if oracle_best_ask is None else oracle_best_ask.get("action"),
                    "oracle_best_ask_family": (
                        None
                        if oracle_best_ask is None
                        else _tool_family(str(oracle_best_ask.get("action", "")))
                    ),
                    "oracle_best_ask_observation_text": (
                        None if oracle_best_ask is None else oracle_best_ask.get("observation_text")
                    ),
                    "oracle_ask_gain": _round_optional(
                        None
                        if oracle_best_ask_eval is None or best_choose_eval is None
                        else oracle_best_ask_eval - best_choose_eval
                    ),
                    "surface_gap": _round_optional(
                        None
                        if oracle_best_ask_eval is None or best_surfaced_ask_eval is None
                        else oracle_best_ask_eval - best_surfaced_ask_eval
                    ),
                    "best_surfaced_ask_missing": best_surfaced_ask_eval is None,
                    "direct_choose_correct": bool(
                        str(step_record.get("action", "")).startswith("CHOOSE_")
                        and task_config is not None
                        and str(step_record.get("action", "")) == f"CHOOSE_{task_config.get('answer')}"
                    ),
                    "blocked_step_class": blocked_step_class,
                    "remaining_scan_action_count": int(step_row.get("remaining_scan_action_count", 0)),
                    "remaining_verify_action_count": int(step_row.get("remaining_verify_action_count", 0)),
                }
            )
        rows_by_mode[mode_name] = flattened
    return rows_by_mode


def _summarize_blocked_step_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = _count_by_key(rows, "blocked_step_class")
    blocked_step_count = len(rows)
    surface_gaps = [
        float(row["surface_gap"])
        for row in rows
        if row.get("surface_gap") is not None
    ]
    task_pressure_rate = _round(_safe_div(class_counts.get("task_pressure", 0), blocked_step_count))
    surfaced_ask_deficit_rate = _round(
        _safe_div(class_counts.get("surfaced_ask_deficit", 0), blocked_step_count)
    )
    hard_cutoff_candidate_rate = _round(
        _safe_div(class_counts.get("hard_cutoff_candidate", 0), blocked_step_count)
    )
    if task_pressure_rate >= BLOCKED_STEP_ROUTE_THRESHOLD:
        route_target = BLOCKED_STEP_CLASS_LABELS["task_pressure"]
    elif surfaced_ask_deficit_rate >= BLOCKED_STEP_ROUTE_THRESHOLD:
        route_target = BLOCKED_STEP_CLASS_LABELS["surfaced_ask_deficit"]
    elif hard_cutoff_candidate_rate >= BLOCKED_STEP_ROUTE_THRESHOLD:
        route_target = BLOCKED_STEP_CLASS_LABELS["hard_cutoff_candidate"]
    else:
        route_target = BLOCKED_STEP_CLASS_LABELS["mixed"]

    return {
        "claim_boundary": BLOCKED_STEP_AUDIT_CLAIM_BOUNDARY,
        "blocked_step_count": blocked_step_count,
        "task_pressure_count": class_counts.get("task_pressure", 0),
        "task_pressure_rate": task_pressure_rate,
        "surfaced_ask_deficit_count": class_counts.get("surfaced_ask_deficit", 0),
        "surfaced_ask_deficit_rate": surfaced_ask_deficit_rate,
        "hard_cutoff_candidate_count": class_counts.get("hard_cutoff_candidate", 0),
        "hard_cutoff_candidate_rate": hard_cutoff_candidate_rate,
        "best_surfaced_ask_missing_count": sum(
            int(bool(row.get("best_surfaced_ask_missing"))) for row in rows
        ),
        "direct_choose_correct_rate": _round(
            _safe_div(sum(int(bool(row.get("direct_choose_correct"))) for row in rows), blocked_step_count)
        ),
        "surface_gap_count": len(surface_gaps),
        "surface_gap_mean": _round_optional(_mean(surface_gaps) if surface_gaps else None),
        "surface_gap_median": _round_optional(_median(surface_gaps) if surface_gaps else None),
        "surface_gap_positive_rate": _round(
            _safe_div(sum(int(value > 1e-6) for value in surface_gaps), len(surface_gaps))
        ),
        "route_target": route_target,
        "route_threshold": BLOCKED_STEP_ROUTE_THRESHOLD,
        "route_requires_new_experiment": route_target == "mixed",
    }


def _group_step_rows(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_value = str(row[key])
        grouped.setdefault(group_value, []).append(row)

    ordered_values = sorted(grouped)
    if key == "step_index":
        ordered_values = sorted(grouped, key=lambda value: int(value))

    return {
        value: _summarize_step_rows(grouped[value])
        for value in ordered_values
    }


def _summarize_z_delta(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in REFLECTION_FIELDS:
        values = [float(episode["z_delta"].get(field, 0.0)) for episode in episodes]
        summary[field] = {
            "mean": _round(_mean(values)),
            "abs_mean": _round(_mean([abs(value) for value in values])),
            "positive_rate": _round(_safe_div(sum(value > 0 for value in values), len(values))),
            "negative_rate": _round(_safe_div(sum(value < 0 for value in values), len(values))),
            "zero_rate": _round(_safe_div(sum(value == 0 for value in values), len(values))),
        }
    return summary


def _avoid_action_bucket(avoid_actions: list[str]) -> str:
    if not avoid_actions:
        return "EMPTY"
    unique = set(avoid_actions)
    if unique == {"CHOOSE"}:
        return "CHOOSE"
    if unique == {"ASK"}:
        return "ASK"
    return "MIXED"


def _summarize_avoid_actions(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets = {"CHOOSE": 0, "ASK": 0, "EMPTY": 0, "MIXED": 0}
    for episode in episodes:
        buckets[_avoid_action_bucket(list(episode["avoid_actions"]))] += 1
    total = len(episodes)
    return {
        name: {"count": count, "rate": _round(_safe_div(count, total))}
        for name, count in buckets.items()
    }


def _subsequent_correlations(episodes: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_seed.setdefault(int(episode["seed"]), []).append(episode)

    for seed_episodes in by_seed.values():
        seed_episodes.sort(key=lambda row: row["task_index"])
        for current, following in zip(seed_episodes, seed_episodes[1:]):
            if not current["reflection_admitted"]:
                continue
            pairs.append((current, following))

    feature_extractors = {
        "avoid_choose_flag": lambda row: float(row["avoid_choose_flag"]),
        "avoid_ask_flag": lambda row: float(row["avoid_ask_flag"]),
        "z_delta.info_seeking": lambda row: float(row["z_delta"]["info_seeking"]),
        "z_delta.cost_sensitivity": lambda row: float(row["z_delta"]["cost_sensitivity"]),
        "z_delta.confidence_threshold": lambda row: float(row["z_delta"]["confidence_threshold"]),
    }
    outcome_extractors = {
        "next_ask_count": lambda row: float(row["ask_count"]),
        "next_steps": lambda row: float(row["steps"]),
        "next_reward": lambda row: float(row["reward"]),
    }

    summary: dict[str, dict[str, float | None]] = {}
    for feature_name, feature_fn in feature_extractors.items():
        xs = [feature_fn(current) for current, _ in pairs]
        summary[feature_name] = {}
        for outcome_name, outcome_fn in outcome_extractors.items():
            ys = [outcome_fn(following) for _, following in pairs]
            summary[feature_name][outcome_name] = _round_optional(_pearson(xs, ys))
    return summary


def _summarize_reflection_bias(
    *,
    episodes: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_ask_rows = [row for row in step_rows if row["selected_type"] == "ASK"]
    regretful_asks = [row for row in selected_ask_rows if row["regretful_ask"]]
    ask_rows_with_retrieval = [row for row in selected_ask_rows if int(row["retrieved_memory_count"]) > 0]

    return {
        "episode_count": len(episodes),
        "reflection_admitted_rate": _round(
            _safe_div(sum(int(episode["reflection_admitted"]) for episode in episodes), len(episodes))
        ),
        "avoid_actions": _summarize_avoid_actions(episodes),
        "z_delta": _summarize_z_delta(episodes),
        "subsequent_correlations": _subsequent_correlations(episodes),
        "memory_alignment": {
            "selected_ask_with_retrieved_memory_rate": _round(
                _safe_div(len(ask_rows_with_retrieval), len(selected_ask_rows))
            ),
            "selected_ask_with_choose_suppressing_memory_rate": _round(
                _safe_div(
                    sum(int(row["retrieved_choose_suppressing_memory"]) for row in ask_rows_with_retrieval),
                    len(ask_rows_with_retrieval),
                )
            ),
            "regretful_ask_with_choose_suppressing_memory_rate": _round(
                _safe_div(
                    sum(int(row["retrieved_choose_suppressing_memory"]) for row in regretful_asks),
                    len(regretful_asks),
                )
            ),
        },
    }


def _summarize_episode_outcomes(episodes: list[dict[str, Any]]) -> dict[str, float]:
    rewards = [float(episode["reward"]) for episode in episodes]
    steps = [float(episode["steps"]) for episode in episodes]
    asks = [float(episode["ask_count"]) for episode in episodes]
    successes = [float(episode["success"]) for episode in episodes]
    return {
        "episode_count": len(episodes),
        "success_rate": _round(_safe_div(sum(successes), len(episodes))),
        "avg_reward": _round(_mean(rewards)),
        "avg_steps": _round(_mean(steps)),
        "avg_asks": _round(_mean(asks)),
    }


def _compare_modes(diagnosis: dict[str, Any], left: str, right: str) -> dict[str, float]:
    left_overall = diagnosis["modes"].get(left, {}).get("overall", {})
    right_overall = diagnosis["modes"].get(right, {}).get("overall", {})
    return {
        "regretful_ask_rate_gap": _round(
            float(left_overall.get("regretful_ask_rate", 0.0))
            - float(right_overall.get("regretful_ask_rate", 0.0))
        ),
        "choose_regret_rate_gap": _round(
            float(left_overall.get("choose_regret_rate", 0.0))
            - float(right_overall.get("choose_regret_rate", 0.0))
        ),
        "selected_ask_rate_gap": _round(
            float(left_overall.get("selected_ask_rate", 0.0))
            - float(right_overall.get("selected_ask_rate", 0.0))
        ),
        "over_ask_cost_mean_gap": _round(
            float(left_overall.get("over_ask_cost_mean_per_step", 0.0))
            - float(right_overall.get("over_ask_cost_mean_per_step", 0.0))
        ),
        "under_ask_cost_mean_gap": _round(
            float(left_overall.get("under_ask_cost_mean_per_step", 0.0))
            - float(right_overall.get("under_ask_cost_mean_per_step", 0.0))
        ),
        "wrong_tool_rate_gap": _round(
            float(left_overall.get("wrong_tool_rate", 0.0))
            - float(right_overall.get("wrong_tool_rate", 0.0))
        ),
        "over_tool_cost_mean_gap": _round(
            float(left_overall.get("over_tool_cost_mean_per_step", 0.0))
            - float(right_overall.get("over_tool_cost_mean_per_step", 0.0))
        ),
        "under_tool_cost_mean_gap": _round(
            float(left_overall.get("under_tool_cost_mean_per_step", 0.0))
            - float(right_overall.get("under_tool_cost_mean_per_step", 0.0))
        ),
    }


def _build_routing(step_rows_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    preferred_modes = [
        mode_name
        for mode_name in ("full", "selector_penalty", "voi_memory_v1")
        if step_rows_by_mode.get(mode_name)
    ]
    routing_modes = preferred_modes or sorted(step_rows_by_mode)
    routing_rows = [
        row
        for mode_name in routing_modes
        for row in step_rows_by_mode.get(mode_name, [])
        if row["oracle_tool_choice_applicable"]
    ]
    mismatch_rows = [
        row
        for row in routing_rows
        if row["selected_vs_oracle_family_mismatch"]
    ]
    choose_swallow_count = sum(
        int(row["selected_vs_oracle_mismatch_kind"] == "choose_swallow")
        for row in mismatch_rows
    )
    surface_family_distortion_count = sum(
        int(row["tool_family_error_source"] == "surface_family_distortion")
        for row in mismatch_rows
    )
    policy_family_error_count = sum(
        int(row["tool_family_error_source"] == "policy_family_error")
        for row in mismatch_rows
    )
    mixed_surface_and_policy_error_count = sum(
        int(row["tool_family_error_source"] == "mixed_surface_and_policy_error")
        for row in mismatch_rows
    )
    candidate_related_share = _round(
        _safe_div(
            choose_swallow_count
            + surface_family_distortion_count
            + mixed_surface_and_policy_error_count,
            len(mismatch_rows),
        )
    )
    policy_family_error_share = _round(
        _safe_div(policy_family_error_count, len(mismatch_rows))
    )
    if candidate_related_share >= 0.60:
        most_likely_layer = "candidate_coverage"
    elif policy_family_error_share > 0.40:
        most_likely_layer = "policy_memory"
    else:
        most_likely_layer = "runtime_backend"

    key_symptoms = [
        f"selected-vs-oracle mismatch rows={len(mismatch_rows)} across modes {', '.join(routing_modes)}",
        f"choose_swallow={choose_swallow_count}, surface_distortion={surface_family_distortion_count}, policy_family_error={policy_family_error_count}",
    ]
    if most_likely_layer == "candidate_coverage":
        key_symptoms.append("candidate-related mismatch sources dominate the routed mismatch pool")
    elif most_likely_layer == "policy_memory":
        key_symptoms.append("policy-family errors dominate enough mismatch rows to justify a policy-memory branch")
    else:
        key_symptoms.append("candidate and policy error shares are both below the routing thresholds")

    return {
        "routing_modes": routing_modes,
        "oracle_tool_row_count": len(routing_rows),
        "selected_vs_oracle_mismatch_count": len(mismatch_rows),
        "candidate_related_share": candidate_related_share,
        "policy_family_error_share": policy_family_error_share,
        "most_likely_layer": most_likely_layer,
        "key_symptoms": key_symptoms,
        "key_metrics": [
            {"name": "candidate_related_share", "value": candidate_related_share},
            {"name": "policy_family_error_share", "value": policy_family_error_share},
            {"name": "selected_vs_oracle_mismatch_count", "value": len(mismatch_rows)},
            {"name": "oracle_tool_row_count", "value": len(routing_rows)},
        ],
        "alternative_explanations": [
            "runtime_backend",
            "policy_memory",
        ] if most_likely_layer == "candidate_coverage" else [
            "candidate_coverage",
            "runtime_backend",
        ],
    }


def _collect_verify_survival_rows(
    step_rows_by_mode: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode_name, step_rows in step_rows_by_mode.items():
        flattened: list[dict[str, Any]] = []
        for row in step_rows:
            for lineage_row in row.get("verify_lineage_audit", []):
                flattened.append(
                    {
                        "seed": row["seed"],
                        "mode": mode_name,
                        "task_id": row["task_id"],
                        "task_index": row["task_index"],
                        "step_index": row["step_index"],
                        "difficulty": row["difficulty"],
                        "noise_level": row["noise_level"],
                        "tool_profile": row["tool_profile"],
                        "tool_subtype": row["tool_subtype"],
                        "candidate_backend": row["candidate_backend"],
                        "ask_gate_enabled": row["ask_gate_enabled"],
                        "ask_gate_allowed": row["ask_gate_allowed"],
                        "best_verify_missing_reason": row["best_verify_missing_reason"],
                        "verify_candidate_exposed": row["verify_candidate_exposed"],
                        **lineage_row,
                    }
                )
        rows_by_mode[mode_name] = flattened
    return rows_by_mode


def _build_recommendation(diagnosis: dict[str, Any]) -> dict[str, Any]:
    modes = diagnosis["modes"]
    full_overall = modes.get("full", {}).get("overall", {})
    full_episode_summary = modes.get("full", {}).get("episodes", {})
    full_bias = diagnosis["reflection_bias"].get("full", {})
    full_correlations = full_bias.get("subsequent_correlations", {})
    full_memory_alignment = full_bias.get("memory_alignment", {})

    choose_corr = full_correlations.get("avoid_choose_flag", {}).get("next_ask_count") or 0.0
    threshold_corr = full_correlations.get("z_delta.confidence_threshold", {}).get("next_ask_count") or 0.0
    choose_memory_rate = float(full_memory_alignment.get("regretful_ask_with_choose_suppressing_memory_rate", 0.0))
    selector_override_rate = float(
        full_overall.get("selector_override_rate_given_nonpositive_ask_advantage", 0.0)
    )
    high_margin_share = float(full_overall.get("regretful_ask_high_margin_share", 0.0))
    easy_share = float(full_overall.get("regretful_ask_easy_share", 0.0))
    low_noise_share = float(full_overall.get("regretful_ask_low_noise_share", 0.0))
    regretful_ask_gap = float(
        diagnosis.get("comparisons", {})
        .get("full_vs_no_reflection", {})
        .get("regretful_ask_rate_gap", 0.0)
    )

    experimental_candidates = [
        mode_name
        for mode_name in EXPERIMENTAL_MODE_ORDER
        if modes.get(mode_name, {}).get("episodes")
    ]
    if experimental_candidates:
        ranked_candidates = sorted(
            experimental_candidates,
            key=lambda mode_name: (
                float(modes[mode_name]["episodes"].get("avg_reward", 0.0)),
                float(modes[mode_name]["episodes"].get("success_rate", 0.0)),
                -float(modes[mode_name]["overall"].get("under_ask_cost_mean_per_step", 0.0)),
                -float(modes[mode_name]["overall"].get("over_ask_cost_mean_per_step", 0.0)),
                -float(modes[mode_name]["overall"].get("under_tool_cost_mean_per_step", 0.0)),
                -float(modes[mode_name]["overall"].get("over_tool_cost_mean_per_step", 0.0)),
                -float(modes[mode_name]["overall"].get("wrong_tool_rate", 0.0)),
                float(modes[mode_name]["overall"].get("tool_precision", 0.0)),
                -float(modes[mode_name]["overall"].get("choose_regret_rate", 0.0)),
                -float(modes[mode_name]["overall"].get("regretful_ask_rate", 0.0)),
            ),
            reverse=True,
        )
        best_mode = ranked_candidates[0]
        best_overall = modes[best_mode]["overall"]
        best_episodes = modes[best_mode]["episodes"]
        reasons = [
            (
                f"{best_mode} has the strongest experimental profile: "
                f"avg_reward={float(best_episodes.get('avg_reward', 0.0)):.3f}, "
                f"success_rate={float(best_episodes.get('success_rate', 0.0)):.3f}."
            ),
            (
                f"{best_mode} keeps regretful_ask_rate="
                f"{float(best_overall.get('regretful_ask_rate', 0.0)):.3f} and "
                f"choose_regret_rate={float(best_overall.get('choose_regret_rate', 0.0)):.3f}."
            ),
            (
                f"{best_mode} keeps over_ask_cost_per_step="
                f"{float(best_overall.get('over_ask_cost_mean_per_step', 0.0)):.3f} and "
                f"under_ask_cost_per_step={float(best_overall.get('under_ask_cost_mean_per_step', 0.0)):.3f}."
            ),
        ]
        if float(best_overall.get("tool_choice_step_count", 0.0)) > 0:
            reasons.append(
                (
                    f"{best_mode} keeps tool_precision={float(best_overall.get('tool_precision', 0.0)):.3f}, "
                    f"wrong_tool_rate={float(best_overall.get('wrong_tool_rate', 0.0)):.3f}, "
                    f"over_tool_cost_per_step={float(best_overall.get('over_tool_cost_mean_per_step', 0.0)):.3f}, "
                    f"under_tool_cost_per_step={float(best_overall.get('under_tool_cost_mean_per_step', 0.0)):.3f}."
                )
            )
        if best_mode != "full" and full_episode_summary:
            reasons.append(
                (
                    f"Compared with full: reward delta="
                    f"{float(best_episodes.get('avg_reward', 0.0)) - float(full_episode_summary.get('avg_reward', 0.0)):.3f}, "
                    f"success delta="
                    f"{float(best_episodes.get('success_rate', 0.0)) - float(full_episode_summary.get('success_rate', 0.0)):.3f}."
                )
            )
        return {
            "recommended_intervention": best_mode,
            "reasons": reasons,
            "decision_rule_inputs": {
                "candidate_modes": ranked_candidates,
                "avg_reward": {
                    mode_name: _round(modes[mode_name]["episodes"].get("avg_reward", 0.0))
                    for mode_name in ranked_candidates
                },
                "success_rate": {
                    mode_name: _round(modes[mode_name]["episodes"].get("success_rate", 0.0))
                    for mode_name in ranked_candidates
                },
                "regretful_ask_rate": {
                    mode_name: _round(modes[mode_name]["overall"].get("regretful_ask_rate", 0.0))
                    for mode_name in ranked_candidates
                },
                "choose_regret_rate": {
                    mode_name: _round(modes[mode_name]["overall"].get("choose_regret_rate", 0.0))
                    for mode_name in ranked_candidates
                },
                "over_ask_cost_mean_per_step": {
                    mode_name: _round(modes[mode_name]["overall"].get("over_ask_cost_mean_per_step", 0.0))
                    for mode_name in ranked_candidates
                },
                "under_ask_cost_mean_per_step": {
                    mode_name: _round(modes[mode_name]["overall"].get("under_ask_cost_mean_per_step", 0.0))
                    for mode_name in ranked_candidates
                },
                "tool_precision": {
                    mode_name: _round(modes[mode_name]["overall"].get("tool_precision", 0.0))
                    for mode_name in ranked_candidates
                },
                "wrong_tool_rate": {
                    mode_name: _round(modes[mode_name]["overall"].get("wrong_tool_rate", 0.0))
                    for mode_name in ranked_candidates
                },
                "over_tool_cost_mean_per_step": {
                    mode_name: _round(modes[mode_name]["overall"].get("over_tool_cost_mean_per_step", 0.0))
                    for mode_name in ranked_candidates
                },
                "under_tool_cost_mean_per_step": {
                    mode_name: _round(modes[mode_name]["overall"].get("under_tool_cost_mean_per_step", 0.0))
                    for mode_name in ranked_candidates
                },
            },
        }

    reasons: list[str] = []
    if selector_override_rate >= 0.5:
        decision = "selector penalty"
        reasons.append(
            f"Selector override is frequent when ask advantage is nonpositive ({selector_override_rate:.3f})."
        )
    elif max(abs(choose_corr), abs(threshold_corr), choose_memory_rate) >= 0.2:
        decision = "verify-then-reflect"
        reasons.append(
            "Regretful asks are materially associated with CHOOSE suppression signals."
        )
    elif (
        regretful_ask_gap > 0
        and easy_share >= 0.45
        and low_noise_share >= 0.45
        and high_margin_share >= 0.25
    ):
        decision = "gating"
        reasons.append(
            "Regretful asks cluster in easy or low-noise states with already-healthy margins."
        )
    else:
        decision = "verify-then-reflect"
        reasons.append(
            "Reflection-linked CHOOSE caution is a better fit than pure trigger frequency."
        )

    reasons.append(f"full vs no_reflection regretful-ask gap: {regretful_ask_gap:.3f}")
    reasons.append(f"avoid CHOOSE -> next ask corr: {choose_corr:.3f}")
    reasons.append(f"confidence threshold -> next ask corr: {threshold_corr:.3f}")
    reasons.append(f"regretful asks with CHOOSE-suppressing memory: {choose_memory_rate:.3f}")

    return {
        "recommended_intervention": decision,
        "reasons": reasons,
        "decision_rule_inputs": {
            "regretful_ask_rate_gap_full_vs_no_reflection": _round(regretful_ask_gap),
            "selector_override_rate_given_nonpositive_ask_advantage": _round(selector_override_rate),
            "avoid_choose_to_next_ask_correlation": _round(choose_corr),
            "confidence_threshold_to_next_ask_correlation": _round(threshold_corr),
            "regretful_ask_with_choose_suppressing_memory_rate": _round(choose_memory_rate),
            "regretful_ask_high_margin_share": _round(high_margin_share),
            "regretful_ask_easy_share": _round(easy_share),
            "regretful_ask_low_noise_share": _round(low_noise_share),
        },
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _render_subgroup_table(
    title: str,
    subgroup_block: dict[str, dict[str, Any]],
    *,
    left_mode: str,
    right_mode: str,
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Group | Full regretful ASK | No reflection regretful ASK | Full choose regret | No reflection choose regret | Full over-ask cost | Full under-ask cost |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    left_groups = subgroup_block.get(left_mode, {})
    right_groups = subgroup_block.get(right_mode, {})
    group_values = sorted(set(left_groups).union(right_groups))
    for group_value in group_values:
        left_summary = left_groups.get(group_value, {})
        right_summary = right_groups.get(group_value, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    group_value,
                    _format_float(left_summary.get("regretful_ask_rate")),
                    _format_float(right_summary.get("regretful_ask_rate")),
                    _format_float(left_summary.get("choose_regret_rate")),
                    _format_float(right_summary.get("choose_regret_rate")),
                    _format_float(left_summary.get("over_ask_cost_mean_per_step")),
                    _format_float(left_summary.get("under_ask_cost_mean_per_step")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _surface_distribution_text(overall: dict[str, Any], family: str) -> str:
    prefix = f"best_{family}"
    return ", ".join(
        [
            f"canonical={_format_float(overall.get(f'{prefix}_canonical_surface_rate'))}",
            f"fixed={_format_float(overall.get(f'{prefix}_fixed_surface_rate'))}",
            f"ask_only={_format_float(overall.get(f'{prefix}_ask_only_surface_rate'))}",
            f"multi_step={_format_float(overall.get(f'{prefix}_multi_step_surface_rate'))}",
        ]
    )


def _missing_reason_distribution_text(overall: dict[str, Any]) -> str:
    return ", ".join(
        [
            f"not_generated={_format_float(overall.get('best_verify_missing_reason_not_generated_rate'))}",
            f"generated_but_pruned={_format_float(overall.get('best_verify_missing_reason_generated_but_pruned_rate'))}",
            f"generated_noncanonical={_format_float(overall.get('best_verify_missing_reason_generated_noncanonical_rate'))}",
            f"dominated_after_scoring={_format_float(overall.get('best_verify_missing_reason_dominated_after_scoring_rate'))}",
        ]
    )


def _verify_stage_distribution_text(summary: dict[str, Any]) -> str:
    keys = ("generated", "family_merge", "normalization_merge", "dedup", "candidate_cap", "pre_selector_filter", "unknown")
    return ", ".join(
        f"{key}={_format_float(summary.get(f'verify_pruned_stage_{key}_rate'))}"
        for key in keys
    )


def _verify_reason_distribution_text(summary: dict[str, Any]) -> str:
    keys = (
        "shape_replaced_by_equivalent",
        "dominated_within_family",
        "dominated_cross_family",
        "deduplicated",
        "budget_cap",
        "filter_rule",
        "unknown",
    )
    return ", ".join(
        f"{key}={_format_float(summary.get(f'verify_pruned_reason_{key}_rate'))}"
        for key in keys
    )


def _verify_stage_summary_text(summary: dict[str, Any], stage_name: str) -> str:
    return (
        f"denominator={summary.get(f'{stage_name}_denominator', 0)}, "
        f"applicable={summary.get(f'{stage_name}_applicable_count', 0)}, "
        f"retained={summary.get(f'{stage_name}_retained_count', 0)}, "
        f"retained_rate={_format_float(summary.get(f'{stage_name}_retained_rate'))}"
    )


def _distribution_text_from_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _family_pre_selector_survival_text(summary: dict[str, Any]) -> str:
    return (
        f"applicable_count={summary.get('pre_selector_applicable_count', 0)}, "
        f"retained_count={summary.get('pre_selector_retained_count', 0)}, "
        f"retained_rate={_format_float(summary.get('pre_selector_retained_rate'))}"
    )


def _secondary_context_comparison_lines(summary: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    comparison = dict(summary.get("secondary_context_comparison") or {})
    for key in PRE_SELECTOR_SECONDARY_CONTEXT_FIELDS:
        block = dict(comparison.get(key) or {})
        lines.append(
            f"- {key}: filter_rule_mean={_format_float(block.get('filter_rule_mean'))}, "
            f"entered_selector_mean={_format_float(block.get('entered_selector_mean'))}"
        )
    return lines


def _render_pre_selector_pruning_audit(diagnosis: dict[str, Any]) -> str:
    run_root = diagnosis["config"]["run_root"]
    seeds = diagnosis["config"]["seeds"]
    lines = [
        "# Pre-Selector Pruning Audit",
        "",
        f"Run root: `{run_root}`",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Focus",
        "",
        "- Current focus: `pre_selector_pruning_audit`",
        "- Execution lane: `discovery`",
        f"- Claim boundary: {PRE_SELECTOR_CLAIM_BOUNDARY}",
        "- Scope: attribution only for gate-facing block reasons on the existing live seed-7 run; no gate-change conclusion is allowed here.",
        "",
        "## Evidence Schema",
        "",
        "- Primary evidence fields:",
    ]
    lines.extend([f"  - {field}" for field in PRE_SELECTOR_PRIMARY_EVIDENCE_FIELDS])
    lines.extend(
        [
            "- Secondary context fields:",
        ]
    )
    lines.extend([f"  - {field}" for field in PRE_SELECTOR_SECONDARY_CONTEXT_FIELDS])

    for mode_name in ("full", "selector_penalty", "voi_memory_v1"):
        summary = diagnosis.get("pre_selector_pruning", {}).get(mode_name, {})
        if not summary:
            continue
        verify_vs_scan = dict(summary.get("verify_vs_scan_pre_selector_survival") or {})
        lines.extend(
            [
                "",
                f"## {mode_name}",
                "",
                "- Claim boundary: block attribution only; this audit does not recommend changing the gate.",
                "- Verify pre-selector summary: "
                f"denominator={summary.get('pre_selector_denominator', 0)}, "
                f"applicable_count={summary.get('pre_selector_applicable_count', 0)}, "
                f"retained_count={summary.get('pre_selector_retained_count', 0)}, "
                f"retained_rate={_format_float(summary.get('pre_selector_retained_rate'))}",
                f"- Filter-rule count: {summary.get('filter_rule_count', 0)}",
                "- Direct gate evidence coverage: "
                f"filter_rule_without_direct_gate_evidence={summary.get('filter_rule_without_direct_gate_evidence', 0)}",
                "- Filter-rule by ask_gate_reason: "
                f"{_distribution_text_from_counts(dict(summary.get('filter_rule_ask_gate_reason_counts') or {}))}",
                "- Filter-rule by tool_subtype: "
                f"{_distribution_text_from_counts(dict(summary.get('filter_rule_tool_subtype_counts') or {}))}",
                "- Verify vs scan pre-selector survival:",
                f"  - verify: {_family_pre_selector_survival_text(dict(verify_vs_scan.get('verify') or {}))}",
                f"  - scan: {_family_pre_selector_survival_text(dict(verify_vs_scan.get('scan') or {}))}",
                "- Secondary context comparison (descriptive only):",
            ]
        )
        lines.extend(_secondary_context_comparison_lines(summary))

    return "\n".join(lines)


def _render_blocked_step_audit(diagnosis: dict[str, Any]) -> str:
    run_root = diagnosis["config"]["run_root"]
    seeds = diagnosis["config"]["seeds"]
    summary = diagnosis.get("blocked_step_audit", {}).get("voi_memory_v1", {})
    lines = [
        "# Blocked ASK Value Audit",
        "",
        f"Run root: `{run_root}`",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Focus",
        "",
        "- Current focus: `blocked_ask_value_audit`",
        "- Execution lane: `discovery`",
        f"- Claim boundary: {BLOCKED_STEP_AUDIT_CLAIM_BOUNDARY}",
        "- Scope: classify step-level strong_negative_voi blocks on the existing live seed-7 run; do not infer a gate change from this audit alone.",
    ]
    if not summary:
        lines.extend(["", "- No blocked-step audit data available."])
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "## `voi_memory_v1` Strong-Negative Blocks",
            "",
            f"- Blocked step count: {summary.get('blocked_step_count', 0)}",
            f"- task_pressure: count={summary.get('task_pressure_count', 0)}, rate={_format_float(summary.get('task_pressure_rate'))}",
            "- surfaced_ask_deficit: "
            f"count={summary.get('surfaced_ask_deficit_count', 0)}, "
            f"rate={_format_float(summary.get('surfaced_ask_deficit_rate'))}",
            "- hard_cutoff_candidate: "
            f"count={summary.get('hard_cutoff_candidate_count', 0)}, "
            f"rate={_format_float(summary.get('hard_cutoff_candidate_rate'))}",
            f"- Direct choose correct rate: {_format_float(summary.get('direct_choose_correct_rate'))}",
            f"- Best surfaced ASK missing count: {summary.get('best_surfaced_ask_missing_count', 0)}",
            "- Surface gap distribution: "
            f"count={summary.get('surface_gap_count', 0)}, "
            f"mean={_format_float(summary.get('surface_gap_mean'))}, "
            f"median={_format_float(summary.get('surface_gap_median'))}, "
            f"positive_rate={_format_float(summary.get('surface_gap_positive_rate'))}",
            f"- Routing threshold: {_format_float(summary.get('route_threshold'))}",
            f"- Suggested next route: {summary.get('route_target', 'unknown')}",
            "- Requires new experiment: "
            f"{'true' if summary.get('route_requires_new_experiment') else 'false'}",
        ]
    )
    return "\n".join(lines)


def _render_pruning_path_audit(diagnosis: dict[str, Any]) -> str:
    run_root = diagnosis["config"]["run_root"]
    seeds = diagnosis["config"]["seeds"]
    routing = diagnosis.get("routing", {})
    lines = [
        "# Pruning-Path Audit",
        "",
        f"Run root: `{run_root}`",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Focus",
        "",
        "- Current focus: `pruning_path_audit`",
        "- Execution lane: `discovery`",
        "- Scope: verify lineage survival only; no memory, gate, selector, task-generator, or benchmark expansion changes are interpreted here.",
        "",
    ]
    for mode_name in ("full", "selector_penalty", "voi_memory_v1"):
        summary = diagnosis.get("verify_survival", {}).get(mode_name, {})
        if not summary:
            continue
        lines.extend(
            [
                f"## {mode_name}",
                "",
                f"- Verify lineage count: {summary.get('verify_lineage_count', 0)}",
                f"- Generated: {_verify_stage_summary_text(summary, 'generated')}",
                f"- Family merge: {_verify_stage_summary_text(summary, 'family_merge')}",
                f"- Normalization merge: {_verify_stage_summary_text(summary, 'normalization_merge')}",
                f"- Dedup: {_verify_stage_summary_text(summary, 'dedup')}",
                f"- Candidate cap (source-side): {_verify_stage_summary_text(summary, 'candidate_cap')}",
                f"- Pre-selector filter: {_verify_stage_summary_text(summary, 'pre_selector_filter')}",
                f"- Selector entry: {_verify_stage_summary_text(summary, 'selector_entry')}",
                f"- Retained before selector rate: {_format_float(summary.get('verify_retained_before_selector_rate'))}",
                f"- Injected before selector: {_format_float(summary.get('verify_injected_before_selector_rate'))}",
                f"- Shape replacement detected: {_format_float(summary.get('verify_shape_replacement_rate'))}",
                f"- Pruned stage distribution: {_verify_stage_distribution_text(summary)}",
                f"- Pruned reason distribution: {_verify_reason_distribution_text(summary)}",
                "",
            ]
        )
    if routing:
        lines.extend(
            [
                "## Routing",
                "",
                f"- Most likely layer: {routing.get('most_likely_layer', 'unknown')}",
                f"- Candidate-related share: {_format_float(routing.get('candidate_related_share'))}",
                f"- Policy-family error share: {_format_float(routing.get('policy_family_error_share'))}",
                "",
            ]
        )
    return "\n".join(lines)


def _render_diagnosis_report(diagnosis: dict[str, Any]) -> str:
    run_root = diagnosis["config"]["run_root"]
    seeds = diagnosis["config"]["seeds"]
    full = diagnosis["modes"].get("full", {}).get("overall", {})
    ctrl = diagnosis["modes"].get("no_reflection", {}).get("overall", {})
    full_episodes = diagnosis["modes"].get("full", {}).get("episodes", {})
    full_bias = diagnosis["reflection_bias"].get("full", {})
    recommendation = diagnosis["recommendation"]
    routing = diagnosis.get("routing", {})

    lines = [
        "# Reflection Failure-Mode Diagnosis",
        "",
        f"Run root: `{run_root}`",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Full Episode Summary",
        "",
        f"- Success rate: {_format_float(full_episodes.get('success_rate'))}",
        f"- Avg reward: {_format_float(full_episodes.get('avg_reward'))}",
        f"- Avg steps: {_format_float(full_episodes.get('avg_steps'))}",
        f"- Avg asks: {_format_float(full_episodes.get('avg_asks'))}",
        "",
        "## Full vs No Reflection",
        "",
        f"- Full regretful ASK rate: {_format_float(full.get('regretful_ask_rate'))}",
        f"- No reflection regretful ASK rate: {_format_float(ctrl.get('regretful_ask_rate'))}",
        f"- Full choose regret rate: {_format_float(full.get('choose_regret_rate'))}",
        f"- No reflection choose regret rate: {_format_float(ctrl.get('choose_regret_rate'))}",
        f"- Full selected ASK rate: {_format_float(full.get('selected_ask_rate'))}",
        f"- No reflection selected ASK rate: {_format_float(ctrl.get('selected_ask_rate'))}",
        f"- Full over-ask cost per step: {_format_float(full.get('over_ask_cost_mean_per_step'))}",
        f"- No reflection over-ask cost per step: {_format_float(ctrl.get('over_ask_cost_mean_per_step'))}",
        f"- Full under-ask cost per step: {_format_float(full.get('under_ask_cost_mean_per_step'))}",
        f"- No reflection under-ask cost per step: {_format_float(ctrl.get('under_ask_cost_mean_per_step'))}",
        f"- Full choose-without-prior-ask rate: {_format_float(full.get('selected_choose_without_prior_ask_rate'))}",
        f"- No reflection choose-without-prior-ask rate: {_format_float(ctrl.get('selected_choose_without_prior_ask_rate'))}",
        "",
    ]

    if float(full.get("selected_tool_count", 0.0)) > 0 or float(ctrl.get("selected_tool_count", 0.0)) > 0:
        lines.extend(
            [
                "## Tool-Use Summary",
                "",
                f"- Full scan usage rate: {_format_float(full.get('scan_usage_rate'))}",
                f"- No reflection scan usage rate: {_format_float(ctrl.get('scan_usage_rate'))}",
                f"- Full verify usage rate: {_format_float(full.get('verify_usage_rate'))}",
                f"- No reflection verify usage rate: {_format_float(ctrl.get('verify_usage_rate'))}",
                f"- Full tool precision: {_format_float(full.get('tool_precision'))} (tool_choice_step_count={int(float(full.get('tool_choice_step_count', 0.0)))})",
                f"- No reflection tool precision: {_format_float(ctrl.get('tool_precision'))} (tool_choice_step_count={int(float(ctrl.get('tool_choice_step_count', 0.0)))})",
                f"- Full wrong-tool rate: {_format_float(full.get('wrong_tool_rate'))}",
                f"- No reflection wrong-tool rate: {_format_float(ctrl.get('wrong_tool_rate'))}",
                f"- Full over-tool cost per step: {_format_float(full.get('over_tool_cost_mean_per_step'))}",
                f"- No reflection over-tool cost per step: {_format_float(ctrl.get('over_tool_cost_mean_per_step'))}",
                f"- Full under-tool cost per step: {_format_float(full.get('under_tool_cost_mean_per_step'))}",
                f"- No reflection under-tool cost per step: {_format_float(ctrl.get('under_tool_cost_mean_per_step'))}",
                "",
            ]
        )
    if float(full.get("oracle_tool_choice_step_count", 0.0)) > 0 or float(ctrl.get("oracle_tool_choice_step_count", 0.0)) > 0:
        lines.extend(
            [
                "## Runtime Tool Diagnosis",
                "",
                f"- Full candidate backend template rate: {_format_float(full.get('candidate_backend_template_rate'))}",
                f"- Full candidate backend union rate: {_format_float(full.get('candidate_backend_union_rate'))}",
                f"- Full oracle tool-choice step count (ask-opportunity proxy): {_format_float(full.get('oracle_tool_choice_step_count'))}",
                f"- Full oracle family gap mean: {_format_float(full.get('oracle_family_gap_mean'))}",
                f"- Full final oracle family alignment rate: {_format_float(full.get('final_oracle_family_alignment_rate'))}",
                f"- Full final oracle scan coverage rate: {_format_float(full.get('final_oracle_scan_coverage_rate'))}",
                f"- Full final oracle verify coverage rate: {_format_float(full.get('final_oracle_verify_coverage_rate'))}",
                f"- Full selector family override rate: {_format_float(full.get('selector_family_override_rate'))}",
                f"- No reflection oracle tool-choice step count (ask-opportunity proxy): {_format_float(ctrl.get('oracle_tool_choice_step_count'))}",
                f"- No reflection final oracle family alignment rate: {_format_float(ctrl.get('final_oracle_family_alignment_rate'))}",
                f"- No reflection selector family override rate: {_format_float(ctrl.get('selector_family_override_rate'))}",
                "",
            ]
        )

    candidate_layer_modes = [
        mode_name
        for mode_name in ("full", "selector_penalty", "voi_memory_v1")
        if diagnosis["modes"].get(mode_name, {}).get("overall")
    ]
    if candidate_layer_modes:
        lines.extend(
            [
                "## Candidate-Layer Diagnosis",
                "",
            ]
        )
        for mode_name in candidate_layer_modes:
            overall = diagnosis["modes"][mode_name]["overall"]
            lines.extend(
                [
                    f"### {mode_name}",
                    "",
                    f"- Selected-vs-oracle mismatch rate: {_format_float(overall.get('selected_vs_oracle_family_mismatch_rate'))}",
                    f"- Choose-swallow rate: {_format_float(overall.get('choose_swallow_rate_given_oracle_tool_choice'))}",
                    f"- Surface-family distortion rate: {_format_float(overall.get('surface_family_distortion_rate'))}",
                    f"- Policy-family error rate: {_format_float(overall.get('policy_family_error_rate'))}",
                    f"- Best-verify canonical surface before: {_format_float(overall.get('best_verify_canonical_surface_rate_before'))}",
                    f"- Best-verify canonical surface after: {_format_float(overall.get('best_verify_canonical_surface_rate_after'))}",
                    f"- Verify candidate exposed rate: {_format_float(overall.get('verify_candidate_exposed_rate'))}",
                    f"- Best-verify missing reasons: {_missing_reason_distribution_text(overall)}",
                    f"- Best-scan surface-shape distribution: {_surface_distribution_text(overall, 'scan')}",
                    f"- Best-verify surface-shape distribution: {_surface_distribution_text(overall, 'verify')}",
                    "",
                ]
            )
    verify_audit_modes = [
        mode_name
        for mode_name in ("full", "selector_penalty", "voi_memory_v1")
        if diagnosis.get("verify_survival", {}).get(mode_name)
    ]
    if verify_audit_modes:
        lines.extend(
            [
                "## Verify Lineage Survival Audit",
                "",
            ]
        )
        for mode_name in verify_audit_modes:
            summary = diagnosis["verify_survival"][mode_name]
            lines.extend(
                [
                    f"### {mode_name}",
                    "",
                    f"- Verify lineage count: {summary.get('verify_lineage_count', 0)}",
                    f"- Generated: {_verify_stage_summary_text(summary, 'generated')}",
                    f"- Family merge: {_verify_stage_summary_text(summary, 'family_merge')}",
                    f"- Normalization merge: {_verify_stage_summary_text(summary, 'normalization_merge')}",
                    f"- Dedup: {_verify_stage_summary_text(summary, 'dedup')}",
                    f"- Candidate cap (source-side): {_verify_stage_summary_text(summary, 'candidate_cap')}",
                    f"- Pre-selector filter: {_verify_stage_summary_text(summary, 'pre_selector_filter')}",
                    f"- Selector entry: {_verify_stage_summary_text(summary, 'selector_entry')}",
                    f"- Retained before selector rate: {_format_float(summary.get('verify_retained_before_selector_rate'))}",
                    f"- Injected before selector: {_format_float(summary.get('verify_injected_before_selector_rate'))}",
                    f"- Shape replacement detected: {_format_float(summary.get('verify_shape_replacement_rate'))}",
                    f"- Pruned stage distribution: {_verify_stage_distribution_text(summary)}",
                    f"- Pruned reason distribution: {_verify_reason_distribution_text(summary)}",
                    "",
                ]
            )
    pre_selector_summary = diagnosis.get("pre_selector_pruning", {}).get("voi_memory_v1", {})
    if pre_selector_summary:
        verify_vs_scan = dict(pre_selector_summary.get("verify_vs_scan_pre_selector_survival") or {})
        lines.extend(
            [
                "## Pre-Selector Pruning Attribution",
                "",
                f"- Claim boundary: {PRE_SELECTOR_CLAIM_BOUNDARY}",
                "- Primary evidence fields:",
            ]
        )
        lines.extend([f"  - {field}" for field in PRE_SELECTOR_PRIMARY_EVIDENCE_FIELDS])
        lines.extend(
            [
                "- Secondary context fields:",
            ]
        )
        lines.extend([f"  - {field}" for field in PRE_SELECTOR_SECONDARY_CONTEXT_FIELDS])
        lines.extend(
            [
                "- `voi_memory_v1` pre-selector summary: "
                f"denominator={pre_selector_summary.get('pre_selector_denominator', 0)}, "
                f"applicable_count={pre_selector_summary.get('pre_selector_applicable_count', 0)}, "
                f"retained_count={pre_selector_summary.get('pre_selector_retained_count', 0)}, "
                f"retained_rate={_format_float(pre_selector_summary.get('pre_selector_retained_rate'))}",
                "- Filter-rule by ask_gate_reason: "
                f"{_distribution_text_from_counts(dict(pre_selector_summary.get('filter_rule_ask_gate_reason_counts') or {}))}",
                "- Filter-rule by tool_subtype: "
                f"{_distribution_text_from_counts(dict(pre_selector_summary.get('filter_rule_tool_subtype_counts') or {}))}",
                "- Verify vs scan pre-selector survival:",
                f"  - verify: {_family_pre_selector_survival_text(dict(verify_vs_scan.get('verify') or {}))}",
                f"  - scan: {_family_pre_selector_survival_text(dict(verify_vs_scan.get('scan') or {}))}",
                "- Direct gate evidence coverage: "
                f"filter_rule_without_direct_gate_evidence={pre_selector_summary.get('filter_rule_without_direct_gate_evidence', 0)}",
                "",
            ]
        )
    blocked_step_summary = diagnosis.get("blocked_step_audit", {}).get("voi_memory_v1", {})
    if blocked_step_summary:
        lines.extend(
            [
                "## Blocked ASK Value Audit",
                "",
                f"- Claim boundary: {BLOCKED_STEP_AUDIT_CLAIM_BOUNDARY}",
                f"- Blocked step count: {blocked_step_summary.get('blocked_step_count', 0)}",
                "- task_pressure: "
                f"count={blocked_step_summary.get('task_pressure_count', 0)}, "
                f"rate={_format_float(blocked_step_summary.get('task_pressure_rate'))}",
                "- surfaced_ask_deficit: "
                f"count={blocked_step_summary.get('surfaced_ask_deficit_count', 0)}, "
                f"rate={_format_float(blocked_step_summary.get('surfaced_ask_deficit_rate'))}",
                "- hard_cutoff_candidate: "
                f"count={blocked_step_summary.get('hard_cutoff_candidate_count', 0)}, "
                f"rate={_format_float(blocked_step_summary.get('hard_cutoff_candidate_rate'))}",
                f"- Direct choose correct rate: {_format_float(blocked_step_summary.get('direct_choose_correct_rate'))}",
                "- Surface gap distribution: "
                f"count={blocked_step_summary.get('surface_gap_count', 0)}, "
                f"mean={_format_float(blocked_step_summary.get('surface_gap_mean'))}, "
                f"median={_format_float(blocked_step_summary.get('surface_gap_median'))}, "
                f"positive_rate={_format_float(blocked_step_summary.get('surface_gap_positive_rate'))}",
                f"- Suggested next route: {blocked_step_summary.get('route_target', 'unknown')}",
                "- Requires new experiment: "
                f"{'true' if blocked_step_summary.get('route_requires_new_experiment') else 'false'}",
                "",
            ]
        )
    if routing:
        lines.extend(
            [
                "## Bottleneck Routing",
                "",
                f"- Routing modes: {', '.join(routing.get('routing_modes', []))}",
                f"- Most likely layer: {routing.get('most_likely_layer', 'unknown')}",
                f"- Candidate-related share: {_format_float(routing.get('candidate_related_share'))}",
                f"- Policy-family error share: {_format_float(routing.get('policy_family_error_share'))}",
                "",
            ]
        )

    for mode_name in EXPERIMENTAL_MODE_ORDER:
        overall = diagnosis["modes"].get(mode_name, {}).get("overall", {})
        episodes = diagnosis["modes"].get(mode_name, {}).get("episodes", {})
        if not overall:
            continue
        lines.extend(
            [
                f"## {mode_name} Snapshot",
                "",
                f"- Success rate: {_format_float(episodes.get('success_rate'))}",
                f"- Avg reward: {_format_float(episodes.get('avg_reward'))}",
                f"- Avg steps: {_format_float(episodes.get('avg_steps'))}",
                f"- Avg asks: {_format_float(episodes.get('avg_asks'))}",
                f"- Regretful ASK rate: {_format_float(overall.get('regretful_ask_rate'))}",
                f"- Choose regret rate: {_format_float(overall.get('choose_regret_rate'))}",
                f"- Over-ask cost per step: {_format_float(overall.get('over_ask_cost_mean_per_step'))}",
                f"- Under-ask cost per step: {_format_float(overall.get('under_ask_cost_mean_per_step'))}",
            ]
        )
        if float(overall.get("selected_tool_count", 0.0)) > 0:
            lines.extend(
                [
                    f"- Scan usage rate: {_format_float(overall.get('scan_usage_rate'))}",
                    f"- Verify usage rate: {_format_float(overall.get('verify_usage_rate'))}",
                    f"- Tool precision: {_format_float(overall.get('tool_precision'))} (tool_choice_step_count={int(float(overall.get('tool_choice_step_count', 0.0)))})",
                    f"- Wrong-tool rate: {_format_float(overall.get('wrong_tool_rate'))}",
                    f"- Over-tool cost per step: {_format_float(overall.get('over_tool_cost_mean_per_step'))}",
                    f"- Under-tool cost per step: {_format_float(overall.get('under_tool_cost_mean_per_step'))}",
                ]
            )
        if float(overall.get("oracle_tool_choice_step_count", 0.0)) > 0:
            lines.extend(
                [
                    f"- Candidate backend template rate: {_format_float(overall.get('candidate_backend_template_rate'))}",
                    f"- Candidate backend union rate: {_format_float(overall.get('candidate_backend_union_rate'))}",
                    f"- Oracle tool-choice step count (ask-opportunity proxy): {_format_float(overall.get('oracle_tool_choice_step_count'))}",
                    f"- Oracle family gap mean: {_format_float(overall.get('oracle_family_gap_mean'))}",
                    f"- Final oracle family alignment rate: {_format_float(overall.get('final_oracle_family_alignment_rate'))}",
                    f"- Final oracle scan coverage rate: {_format_float(overall.get('final_oracle_scan_coverage_rate'))}",
                    f"- Final oracle verify coverage rate: {_format_float(overall.get('final_oracle_verify_coverage_rate'))}",
                    f"- Selector family override rate: {_format_float(overall.get('selector_family_override_rate'))}",
                ]
            )
        if float(overall.get("ask_gate_step_count", 0)) > 0:
            lines.extend(
                [
                    f"- Ask-gate block rate: {_format_float(overall.get('ask_gate_block_rate'))}",
                    f"- Ask-gate allow rate given positive ask advantage: {_format_float(overall.get('ask_gate_allow_rate_given_positive_ask_advantage'))}",
                    f"- Ask-gate block rate given nonpositive ask advantage: {_format_float(overall.get('ask_gate_block_rate_given_nonpositive_ask_advantage'))}",
                    f"- Retrieved positive evidence rate: {_format_float(overall.get('retrieved_positive_evidence_rate'))}",
                    f"- Retrieved caution evidence rate: {_format_float(overall.get('retrieved_caution_evidence_rate'))}",
                ]
            )
        lines.append("")

    lines.extend(
        _render_subgroup_table(
            "Difficulty Breakdown",
            diagnosis["subgroups"]["difficulty"],
            left_mode="full",
            right_mode="no_reflection",
        )
    )
    lines.extend(
        _render_subgroup_table(
            "Noise Breakdown",
            diagnosis["subgroups"]["noise_level"],
            left_mode="full",
            right_mode="no_reflection",
        )
    )

    avoid_actions = full_bias.get("avoid_actions", {})
    z_delta = full_bias.get("z_delta", {})
    lines.extend(
        [
            "## Reflection Avoid-Action Distribution",
            "",
            f"- avoid_actions=['CHOOSE']: {_format_float(avoid_actions.get('CHOOSE', {}).get('rate'))} ({avoid_actions.get('CHOOSE', {}).get('count', 0)})",
            f"- avoid_actions=['ASK']: {_format_float(avoid_actions.get('ASK', {}).get('rate'))} ({avoid_actions.get('ASK', {}).get('count', 0)})",
            f"- avoid_actions=[]: {_format_float(avoid_actions.get('EMPTY', {}).get('rate'))} ({avoid_actions.get('EMPTY', {}).get('count', 0)})",
            f"- mixed avoid actions: {_format_float(avoid_actions.get('MIXED', {}).get('rate'))} ({avoid_actions.get('MIXED', {}).get('count', 0)})",
            "",
            "## Reflection Delta Summary",
            "",
        ]
    )

    for field in REFLECTION_FIELDS:
        block = z_delta.get(field, {})
        lines.append(
            f"- {field}: mean={_format_float(block.get('mean'))}, abs_mean={_format_float(block.get('abs_mean'))}, positive={_format_float(block.get('positive_rate'))}, negative={_format_float(block.get('negative_rate'))}"
        )

    correlations = full_bias.get("subsequent_correlations", {})
    memory_alignment = full_bias.get("memory_alignment", {})
    lines.extend(
        [
            "",
            "## Subsequent Correlation Summary",
            "",
            f"- avoid CHOOSE -> next ask count: {_format_float(correlations.get('avoid_choose_flag', {}).get('next_ask_count'))}",
            f"- avoid CHOOSE -> next reward: {_format_float(correlations.get('avoid_choose_flag', {}).get('next_reward'))}",
            f"- confidence threshold delta -> next ask count: {_format_float(correlations.get('z_delta.confidence_threshold', {}).get('next_ask_count'))}",
            f"- info seeking delta -> next ask count: {_format_float(correlations.get('z_delta.info_seeking', {}).get('next_ask_count'))}",
            "",
            "## Memory-Mediated CHOOSE Suppression",
            "",
            f"- Selected ASK with retrieved CHOOSE-suppressing memory: {_format_float(memory_alignment.get('selected_ask_with_choose_suppressing_memory_rate'))}",
            f"- Regretful ASK with CHOOSE-suppressing memory: {_format_float(memory_alignment.get('regretful_ask_with_choose_suppressing_memory_rate'))}",
            "",
            "## Under-asking Profile",
            "",
            f"- Full choose regret rate: {_format_float(full.get('choose_regret_rate'))}",
            f"- Full under-ask cost per choose regret: {_format_float(full.get('under_ask_cost_mean_per_choose_regret'))}",
            f"- Full choose regret without prior ask rate: {_format_float(full.get('choose_regret_without_prior_ask_rate'))}",
            f"- Full choose regret high-margin share: {_format_float(full.get('choose_regret_high_margin_share'))}",
            "",
            "## Recommendation",
            "",
            f"- Recommended next intervention: `{recommendation['recommended_intervention']}`",
        ]
    )
    for reason in recommendation["reasons"]:
        lines.append(f"- {reason}")

    lines.append("")
    return "\n".join(lines)


def run_diagnosis(
    *,
    run_root: Path,
    seeds: list[int] | None = None,
) -> tuple[dict[str, Any], str]:
    detected_seeds, seed_roots = _detect_seeds(
        run_root,
        ",".join(str(seed) for seed in seeds) if seeds is not None else None,
    )

    step_rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    episode_rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    step_contexts_by_mode: dict[str, list[dict[str, Any]]] = {}

    for seed in detected_seeds:
        seed_root = seed_roots[seed]
        if not seed_root.exists():
            raise FileNotFoundError(f"Missing seed directory: {seed_root}")

        for mode_name, spec in MODE_SPECS.items():
            records_path = seed_root / spec["records"]
            if not records_path.exists():
                continue
            memory_index = _load_memory_index(seed_root / spec["memory"])
            records = _load_jsonl(records_path)
            task_index = _load_task_index_for_records(records_path, records)
            prior_actions_by_task: dict[str, list[str]] = {}
            step_rows: list[dict[str, Any]] = []
            for record in records:
                if record.get("record_type") != "step":
                    continue
                task_id = str(record.get("task_id"))
                step_contexts_by_mode.setdefault(mode_name, []).append(
                    {
                        "seed": seed,
                        "mode": mode_name,
                        "step_record": record,
                        "task_config": task_index.get(task_id),
                        "prior_actions": list(prior_actions_by_task.get(task_id, [])),
                    }
                )
                step_rows.append(
                    _analyze_step_record(
                        seed=seed,
                        mode_name=mode_name,
                        step_record=record,
                        memory_index=memory_index,
                        task_config=task_index.get(task_id),
                        prior_actions=prior_actions_by_task.get(task_id, []),
                    )
                )
                prior_actions_by_task.setdefault(task_id, []).append(str(record.get("action", "")))
            episode_rows = _episode_rows_for_records(
                seed=seed,
                mode_name=mode_name,
                records=records,
            )

            step_rows_by_mode.setdefault(mode_name, []).extend(step_rows)
            episode_rows_by_mode.setdefault(mode_name, []).extend(episode_rows)

    verify_survival_rows_by_mode = _collect_verify_survival_rows(step_rows_by_mode)
    pre_selector_pruning_rows_by_mode = _collect_pre_selector_pruning_rows(step_rows_by_mode)
    blocked_step_audit_rows_by_mode = _collect_blocked_step_audit_rows(
        step_contexts_by_mode,
        step_rows_by_mode,
    )

    diagnosis: dict[str, Any] = {
        "config": {
            "run_root": str(run_root),
            "seeds": detected_seeds,
            "seed_count": len(detected_seeds),
        },
        "modes": {},
        "verify_survival": {},
        "pre_selector_pruning": {},
        "blocked_step_audit": {},
        "subgroups": {
            "difficulty": {},
            "noise_level": {},
            "step_index": {},
            "tool_profile": {},
            "tool_subtype": {},
        },
        "reflection_bias": {},
        "comparisons": {},
    }

    for mode_name, rows in step_rows_by_mode.items():
        diagnosis["modes"][mode_name] = {
            "overall": _summarize_step_rows(rows),
        }
        for group_key in ("difficulty", "noise_level", "step_index", "tool_profile", "tool_subtype"):
            diagnosis["subgroups"][group_key][mode_name] = _group_step_rows(rows, group_key)
    for mode_name, rows in verify_survival_rows_by_mode.items():
        diagnosis["verify_survival"][mode_name] = _summarize_verify_survival_rows(rows)
    for mode_name, rows in pre_selector_pruning_rows_by_mode.items():
        diagnosis["pre_selector_pruning"][mode_name] = _summarize_pre_selector_pruning(
            step_rows=step_rows_by_mode.get(mode_name, []),
            verify_survival_summary=diagnosis["verify_survival"].get(mode_name, {}),
            pre_selector_rows=rows,
        )
    for mode_name, rows in blocked_step_audit_rows_by_mode.items():
        diagnosis["blocked_step_audit"][mode_name] = _summarize_blocked_step_audit(rows)

    for mode_name, episodes in episode_rows_by_mode.items():
        diagnosis["modes"].setdefault(mode_name, {})
        diagnosis["modes"][mode_name]["episodes"] = _summarize_episode_outcomes(episodes)
        diagnosis["reflection_bias"][mode_name] = _summarize_reflection_bias(
            episodes=episodes,
            step_rows=step_rows_by_mode.get(mode_name, []),
        )

    diagnosis["comparisons"]["full_vs_no_reflection"] = _compare_modes(
        diagnosis,
        "full",
        "no_reflection",
    )
    diagnosis["comparisons"]["full_vs_permute"] = _compare_modes(
        diagnosis,
        "full",
        "permute",
    )
    diagnosis["_debug_step_rows"] = step_rows_by_mode
    diagnosis["_debug_verify_survival_rows"] = verify_survival_rows_by_mode
    diagnosis["_debug_pre_selector_pruning_rows"] = pre_selector_pruning_rows_by_mode
    diagnosis["_debug_blocked_step_audit_rows"] = blocked_step_audit_rows_by_mode
    diagnosis["routing"] = _build_routing(step_rows_by_mode)
    diagnosis["recommendation"] = _build_recommendation(diagnosis)

    report = _render_diagnosis_report(diagnosis)
    return diagnosis, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose reflection-induced failure modes from run logs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_explicit_seeds(args.seeds) if args.seeds else None
    diagnosis, report = run_diagnosis(run_root=args.run_root, seeds=seeds)
    debug_step_rows = diagnosis.pop("_debug_step_rows", {})
    debug_verify_survival_rows = diagnosis.pop("_debug_verify_survival_rows", {})
    debug_pre_selector_pruning_rows = diagnosis.pop("_debug_pre_selector_pruning_rows", {})
    debug_blocked_step_audit_rows = diagnosis.pop("_debug_blocked_step_audit_rows", {})
    pruning_audit = _render_pruning_path_audit(diagnosis)
    pre_selector_audit = _render_pre_selector_pruning_audit(diagnosis)
    blocked_step_audit = _render_blocked_step_audit(diagnosis)

    step_rows_path = args.run_root / "diagnosis_step_rows.jsonl"
    with step_rows_path.open("w", encoding="utf-8") as handle:
        for mode_name in sorted(debug_step_rows):
            for row in debug_step_rows[mode_name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    verify_survival_rows_path = args.run_root / "verify_survival_rows.jsonl"
    with verify_survival_rows_path.open("w", encoding="utf-8") as handle:
        for mode_name in sorted(debug_verify_survival_rows):
            for row in debug_verify_survival_rows[mode_name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    pre_selector_rows_path = args.run_root / "pre_selector_pruning_rows.jsonl"
    with pre_selector_rows_path.open("w", encoding="utf-8") as handle:
        for mode_name in sorted(debug_pre_selector_pruning_rows):
            for row in debug_pre_selector_pruning_rows[mode_name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    blocked_step_rows_path = args.run_root / "blocked_step_audit_rows.jsonl"
    with blocked_step_rows_path.open("w", encoding="utf-8") as handle:
        for mode_name in sorted(debug_blocked_step_audit_rows):
            for row in debug_blocked_step_audit_rows[mode_name]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.run_root / "diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.run_root / "diagnosis_report.md").write_text(report, encoding="utf-8")
    (args.run_root / "pruning_path_audit.md").write_text(pruning_audit, encoding="utf-8")
    (args.run_root / "pre_selector_pruning_audit.md").write_text(
        pre_selector_audit,
        encoding="utf-8",
    )
    (args.run_root / "blocked_step_audit.md").write_text(
        blocked_step_audit,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
