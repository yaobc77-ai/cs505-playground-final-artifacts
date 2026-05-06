from __future__ import annotations

from typing import Any

DECISION_LESSON_TYPES = (
    "ask_regretful",
    "ask_helpful",
    "choose_regretful",
    "choose_helpful",
)
POSITIVE_LESSON_TYPES = ("ask_helpful", "choose_regretful")
CAUTION_LESSON_TYPES = ("ask_regretful", "choose_helpful")


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def action_tool_family(action: str | None) -> str:
    action = str(action or "")
    if action.startswith("ASK_SCAN_"):
        return "scan"
    if action.startswith("ASK_VERIFY_"):
        return "verify"
    if action.startswith("ASK_CLUE_"):
        return "clue"
    if action.startswith("CHOOSE_"):
        return "choose"
    return "other"


def _task_tags(task: dict[str, Any], mode: str) -> list[str]:
    tags = [
        f"difficulty:{task.get('difficulty', 'unknown')}",
        f"noise:{task.get('noise_level', 0.0)}",
        f"mode:{mode}",
    ]
    if task.get("benchmark_family"):
        tags.append(f"benchmark:{task['benchmark_family']}")
    if task.get("arc_id"):
        tags.append(f"arc:{task['arc_id']}")
    if task.get("episode_index") is not None:
        tags.append(f"episode:{task['episode_index']}")
    return tags


def margin_bucket(margin: float) -> str:
    if margin < 0.25:
        return "lt_0p25"
    if margin < 0.75:
        return "0p25_to_0p75"
    if margin < 1.50:
        return "0p75_to_1p50"
    return "ge_1p50"


def remaining_asks_bucket(remaining_asks: int) -> str:
    if remaining_asks <= 0:
        return "0"
    if remaining_asks == 1:
        return "1"
    return "2plus"


def asked_count_bucket(asked_count: int) -> str:
    if asked_count <= 0:
        return "0"
    if asked_count == 1:
        return "1"
    return "2plus"


def noise_bucket(noise_level: Any) -> str:
    try:
        return f"{float(noise_level):.2f}"
    except (TypeError, ValueError):
        return str(noise_level)


def reliability_bucket(mean_reliability: Any) -> str:
    try:
        value = float(mean_reliability)
    except (TypeError, ValueError):
        return str(mean_reliability)
    if value < 0.55:
        return "lt_0p55"
    if value < 0.80:
        return "0p55_to_0p80"
    return "ge_0p80"


def conflict_bucket(evidence_conflict: Any) -> str:
    try:
        value = float(evidence_conflict)
    except (TypeError, ValueError):
        return str(evidence_conflict)
    if value <= 0.0:
        return "none"
    if value < 0.34:
        return "lt_0p34"
    if value < 0.67:
        return "0p34_to_0p67"
    return "ge_0p67"


def build_decision_scope(
    *,
    task: dict[str, Any],
    belief: dict[str, Any],
    tool_family: str | None = None,
) -> dict[str, Any]:
    remaining_asks = len(list(belief.get("remaining_asks", [])))
    asked_count = int(belief.get("asked_count", 0))
    margin = float(belief.get("margin", 0.0))
    mean_reliability = float(belief.get("mean_reliability", 0.0))
    evidence_conflict = float(belief.get("evidence_conflict", 0.0))
    return {
        "difficulty": str(task.get("difficulty", "unknown")),
        "noise_bucket": noise_bucket(task.get("noise_level", 0.0)),
        "margin_bucket": margin_bucket(margin),
        "remaining_asks_bucket": remaining_asks_bucket(remaining_asks),
        "asked_count_bucket": asked_count_bucket(asked_count),
        "tool_family": str(tool_family or "any"),
        "reliability_bucket": reliability_bucket(mean_reliability),
        "conflict_bucket": conflict_bucket(evidence_conflict),
        "margin": round(margin, 4),
        "remaining_asks": remaining_asks,
        "asked_count": asked_count,
        "mean_reliability": round(mean_reliability, 4),
        "evidence_conflict": round(evidence_conflict, 4),
    }


def decision_lesson_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str, str]:
    return (
        str(entry.get("lesson_type", "")),
        str(entry.get("difficulty", "")),
        str(entry.get("noise_bucket", "")),
        str(entry.get("margin_bucket", "")),
        str(entry.get("remaining_asks_bucket", "")),
        str(entry.get("asked_count_bucket", "")),
        str(entry.get("tool_family", "")),
        str(entry.get("reliability_bucket", "")),
        str(entry.get("conflict_bucket", "")),
    )


def classify_decision_lesson(
    *,
    selected_action: str,
    ask_advantage: float | None,
    ask_cost: float,
) -> str | None:
    if ask_advantage is None:
        return None

    cost = max(abs(float(ask_cost)), 1e-6)
    if abs(float(ask_advantage)) / cost < 0.25:
        return None

    if selected_action.startswith("ASK"):
        if ask_advantage <= 0:
            return "ask_regretful"
        if ask_advantage >= 0.50 * cost:
            return "ask_helpful"
        return None

    if selected_action.startswith("CHOOSE"):
        if ask_advantage >= 0.25 * cost:
            return "choose_regretful"
        if ask_advantage <= -0.50 * cost:
            return "choose_helpful"
        return None

    return None


def build_decision_lesson(
    *,
    task: dict[str, Any],
    mode: str,
    belief: dict[str, Any],
    selected_action: str,
    selected_candidate_name: str,
    ask_advantage: float | None,
    ask_cost: float,
    reward: float,
    step_index: int,
    best_ask_family: str | None = None,
) -> dict[str, Any] | None:
    lesson_type = classify_decision_lesson(
        selected_action=selected_action,
        ask_advantage=ask_advantage,
        ask_cost=ask_cost,
    )
    if lesson_type is None or ask_advantage is None:
        return None

    selected_family = action_tool_family(selected_action)
    lesson_tool_family = selected_family if selected_family != "choose" else str(best_ask_family or "none")
    scope = build_decision_scope(task=task, belief=belief, tool_family=lesson_tool_family)
    cost = max(abs(float(ask_cost)), 1e-6)
    normalized_advantage = min(1.0, abs(float(ask_advantage)) / cost)
    strength = round(clamp(normalized_advantage, 0.0, 1.0), 4)
    content = (
        f"{lesson_type} under difficulty={scope['difficulty']} noise={scope['noise_bucket']} "
        f"margin_bucket={scope['margin_bucket']} remaining_asks_bucket={scope['remaining_asks_bucket']} "
        f"asked_count_bucket={scope['asked_count_bucket']} tool_family={scope['tool_family']} "
        f"reliability_bucket={scope['reliability_bucket']} conflict_bucket={scope['conflict_bucket']} "
        f"selected_action={selected_action} "
        f"ask_advantage={float(ask_advantage):.4f}"
    )

    return {
        "type": "decision_lesson",
        "lesson_type": lesson_type,
        "difficulty": scope["difficulty"],
        "noise_bucket": scope["noise_bucket"],
        "margin_bucket": scope["margin_bucket"],
        "remaining_asks_bucket": scope["remaining_asks_bucket"],
        "asked_count_bucket": scope["asked_count_bucket"],
        "tool_family": scope["tool_family"],
        "reliability_bucket": scope["reliability_bucket"],
        "conflict_bucket": scope["conflict_bucket"],
        "strength": strength,
        "reinforcement_count": 1,
        "last_seen_version": 0,
        "task_tags": _task_tags(task, mode),
        "provenance": {
            "task_id": task["task_id"],
            "mode": mode,
            "step": step_index,
            "selected_action": selected_action,
            "selected_candidate_name": selected_candidate_name,
            "tool_family": lesson_tool_family,
            "ask_advantage": round(float(ask_advantage), 4),
            "reward": round(float(reward), 4),
        },
        "content": content[:400],
    }


def recency_bonus(*, current_version: int, last_seen_version: int) -> float:
    age_in_versions = max(0, int(current_version) - int(last_seen_version))
    return 1.0 / (1.0 + age_in_versions / 25.0)


def retention_score(entry: dict[str, Any], *, current_version: int) -> float:
    seen_version = int(entry.get("last_seen_version", entry.get("version", 0)) or 0)
    bonus = recency_bonus(current_version=current_version, last_seen_version=seen_version)
    return (
        0.6 * float(entry.get("strength", 0.0))
        + 0.25 * min(float(entry.get("reinforcement_count", 0)) / 5.0, 1.0)
        + 0.15 * bonus
    )


def _scope_match(entry_value: str, query_value: str) -> float:
    return 1.0 if str(entry_value) == str(query_value) else 0.0


def decision_lesson_match_score(
    entry: dict[str, Any],
    *,
    query_scope: dict[str, Any],
    current_version: int,
) -> float:
    margin_match = _scope_match(entry.get("margin_bucket", ""), query_scope.get("margin_bucket", ""))
    difficulty_match = _scope_match(entry.get("difficulty", ""), query_scope.get("difficulty", ""))
    noise_match = _scope_match(entry.get("noise_bucket", ""), query_scope.get("noise_bucket", ""))
    remaining_asks_match = _scope_match(
        entry.get("remaining_asks_bucket", ""),
        query_scope.get("remaining_asks_bucket", ""),
    )
    asked_count_match = _scope_match(
        entry.get("asked_count_bucket", ""),
        query_scope.get("asked_count_bucket", ""),
    )
    tool_family_match = _scope_match(entry.get("tool_family", ""), query_scope.get("tool_family", ""))
    reliability_match = _scope_match(
        entry.get("reliability_bucket", ""),
        query_scope.get("reliability_bucket", ""),
    )
    conflict_match = _scope_match(
        entry.get("conflict_bucket", ""),
        query_scope.get("conflict_bucket", ""),
    )
    bonus = recency_bonus(
        current_version=current_version,
        last_seen_version=int(entry.get("last_seen_version", entry.get("version", 0)) or 0),
    )
    return (
        2.0 * margin_match
        + 1.5 * difficulty_match
        + 1.0 * noise_match
        + 1.0 * remaining_asks_match
        + 0.5 * asked_count_match
        + 1.25 * tool_family_match
        + 1.0 * reliability_match
        + 0.75 * conflict_match
        + 1.5 * float(entry.get("strength", 0.0))
        + 0.5 * bonus
    )


def summarize_decision_lesson_signal(
    *,
    positive_lessons: list[dict[str, Any]],
    caution_lessons: list[dict[str, Any]],
) -> dict[str, float]:
    bucket_totals = {
        "helpful_ask": 0.0,
        "choose_regret": 0.0,
        "regretful_ask": 0.0,
        "choose_helpful": 0.0,
    }
    for entry in positive_lessons + caution_lessons:
        strength = float(entry.get("strength", 0.0))
        lesson_type = str(entry.get("lesson_type", ""))
        if lesson_type == "ask_helpful":
            bucket_totals["helpful_ask"] += strength
        elif lesson_type == "choose_regretful":
            bucket_totals["choose_regret"] += strength
        elif lesson_type == "ask_regretful":
            bucket_totals["regretful_ask"] += strength
        elif lesson_type == "choose_helpful":
            bucket_totals["choose_helpful"] += strength
    return {
        key: round(min(1.0, value), 4)
        for key, value in bucket_totals.items()
    }


def merge_voi_candidates(
    *,
    llm_candidates: list[dict[str, Any]] | None,
    template_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for source_candidates in (llm_candidates or [], template_candidates):
        for candidate in source_candidates:
            actions = tuple(str(action) for action in candidate.get("actions", []))
            if not actions or actions in seen:
                continue
            seen.add(actions)
            merged.append(dict(candidate))
    return merged


def _best_eval(
    candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    action_prefix: str,
) -> tuple[float | None, str | None]:
    best: float | None = None
    best_family: str | None = None
    for candidate, evaluation in zip(candidates, evaluations, strict=True):
        actions = list(candidate.get("actions", []))
        if not actions or not str(actions[0]).startswith(action_prefix):
            continue
        score = float(evaluation.get("score", 0.0))
        if best is None or score > best:
            best = score
            best_family = action_tool_family(actions[0])
    return best, best_family


def evaluate_ask_gate(
    *,
    candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    belief: dict[str, Any],
    confidence_threshold: float,
    ask_cost: float,
    memory_retrieval: dict[str, Any] | None,
) -> dict[str, Any]:
    memory_retrieval = dict(memory_retrieval or {})
    best_ask_eval, best_ask_family = _best_eval(candidates, evaluations, "ASK")
    best_choose_eval, _ = _best_eval(candidates, evaluations, "CHOOSE")
    ask_advantage = (
        best_ask_eval - best_choose_eval
        if best_ask_eval is not None and best_choose_eval is not None
        else None
    )
    cost = max(abs(float(ask_cost)), 1e-6)
    margin = float(belief.get("margin", 0.0))
    threshold = max(float(confidence_threshold), 1e-6)
    base_uncertainty = clamp((threshold - margin) / threshold, 0.0, 1.0)
    mean_reliability = float(belief.get("mean_reliability", 0.0))
    evidence_conflict = clamp(float(belief.get("evidence_conflict", 0.0)), 0.0, 1.0)
    reliability_uncertainty = clamp((0.80 - mean_reliability) / 0.80, 0.0, 1.0)
    uncertainty = clamp(
        base_uncertainty + 0.25 * reliability_uncertainty + 0.35 * evidence_conflict,
        0.0,
        1.0,
    )
    family_summary = {}
    if best_ask_family:
        family_summary = dict(
            (
                memory_retrieval.get("by_family", {}) or {}
            ).get(best_ask_family, {}).get("summary", {})
        )
    memory_summary = family_summary or dict(memory_retrieval.get("summary", {}))
    helpful_ask = float(memory_summary.get("helpful_ask", 0.0))
    choose_regret = float(memory_summary.get("choose_regret", 0.0))
    regretful_ask = float(memory_summary.get("regretful_ask", 0.0))
    choose_helpful = float(memory_summary.get("choose_helpful", 0.0))
    memory_signal = helpful_ask + choose_regret - regretful_ask - choose_helpful

    result = {
        "allowed": True,
        "score": None,
        "reason": "bypass_missing_ask_or_choose",
        "inputs": {
            "best_ask_eval": round(best_ask_eval, 4) if best_ask_eval is not None else None,
            "best_choose_eval": round(best_choose_eval, 4) if best_choose_eval is not None else None,
            "best_ask_family": best_ask_family,
            "ask_advantage": round(ask_advantage, 4) if ask_advantage is not None else None,
            "ask_cost": round(cost, 4),
            "voi_ratio": round(ask_advantage / cost, 4) if ask_advantage is not None else None,
            "margin": round(margin, 4),
            "confidence_threshold": round(float(confidence_threshold), 4),
            "base_uncertainty": round(base_uncertainty, 4),
            "uncertainty": round(uncertainty, 4),
            "mean_reliability": round(mean_reliability, 4),
            "reliability_uncertainty": round(reliability_uncertainty, 4),
            "evidence_conflict": round(evidence_conflict, 4),
            "memory_signal": round(memory_signal, 4),
            "helpful_ask": round(helpful_ask, 4),
            "choose_regret": round(choose_regret, 4),
            "regretful_ask": round(regretful_ask, 4),
            "choose_helpful": round(choose_helpful, 4),
        },
    }
    if ask_advantage is None:
        return result

    voi_ratio = ask_advantage / cost
    result["inputs"]["voi_ratio"] = round(voi_ratio, 4)
    if ask_advantage >= 0.50 * cost:
        result["reason"] = "strong_positive_voi"
        return result
    if ask_advantage <= -0.25 * cost:
        result["allowed"] = False
        result["reason"] = "strong_negative_voi"
        return result

    score = 1.0 * voi_ratio + 0.6 * uncertainty + 0.5 * memory_signal
    result["score"] = round(score, 4)
    result["allowed"] = score >= 0.35
    result["reason"] = "borderline_allow" if result["allowed"] else "borderline_block"
    return result
