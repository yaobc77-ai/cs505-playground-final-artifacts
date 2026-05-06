from __future__ import annotations

from typing import Any

from agent.counterfactual import belief_from_observation
from agent.recognition import assess_candidate_recognition
from agent.self_model import SelfModel

DEFAULT_SELECTOR_PENALTY_BASE = 0.10
DEFAULT_SELECTOR_PENALTY_MEMORY_SCALE = 0.45


def _matches_avoid(action: str, avoid_action: str) -> bool:
    if avoid_action in {"ASK", "CHOOSE"}:
        return action.startswith(avoid_action)
    return action == avoid_action


def _memory_penalty(
    action: str,
    memory_entries: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    penalty = 0.0
    matched_ids: list[str] = []
    for entry in memory_entries:
        for avoid_action in entry.get("avoid_actions", []):
            if _matches_avoid(action, avoid_action):
                penalty -= 1.0
                matched_ids.append(str(entry.get("memory_id", "unknown")))
                break
    return penalty, matched_ids


def _suppresses_choose(entry: dict[str, Any]) -> bool:
    return any(
        avoid_action == "CHOOSE" or str(avoid_action).startswith("CHOOSE_")
        for avoid_action in entry.get("avoid_actions", [])
    )


def _preference_score(
    action: str,
    self_model: SelfModel,
    observation: dict[str, Any],
) -> float:
    belief = belief_from_observation(observation)
    margin = belief["margin"]

    if action.startswith("ASK"):
        return 0.9 * self_model.info_seeking - 0.5 * self_model.cost_sensitivity

    chosen = action.removeprefix("CHOOSE_")
    score = 0.8 * self_model.cost_sensitivity
    if chosen == belief["top_choice"]:
        score += 0.3
    else:
        score -= 0.4

    if margin < self_model.confidence_threshold:
        score -= self_model.confidence_threshold - margin
    else:
        score += 0.2 * margin
    return score


def select_candidate(
    observation: dict[str, Any],
    candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    self_model: SelfModel,
    memory_entries: list[dict[str, Any]],
    selector_mode: str = "default",
    selector_config: dict[str, Any] | None = None,
    subject_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selector_config = dict(selector_config or {})
    subject_context = dict(subject_context or {})
    selector_penalty_base = max(
        0.0,
        float(selector_config.get("penalty_base", DEFAULT_SELECTOR_PENALTY_BASE)),
    )
    selector_penalty_memory_scale = max(
        0.0,
        float(
            selector_config.get(
                "penalty_memory_scale",
                DEFAULT_SELECTOR_PENALTY_MEMORY_SCALE,
            )
        ),
    )
    evaluation_by_name = {
        evaluation["candidate_name"]: evaluation for evaluation in evaluations
    }
    best_ask_eval = max(
        (
            float(evaluation["score"])
            for candidate, evaluation in zip(candidates, evaluations, strict=True)
            if str(candidate["actions"][0]).startswith("ASK")
        ),
        default=None,
    )
    best_choose_eval = max(
        (
            float(evaluation["score"])
            for candidate, evaluation in zip(candidates, evaluations, strict=True)
            if str(candidate["actions"][0]).startswith("CHOOSE")
        ),
        default=None,
    )
    ask_advantage = (
        best_ask_eval - best_choose_eval
        if best_ask_eval is not None and best_choose_eval is not None
        else None
    )
    choose_suppressing_memory_count = sum(1 for entry in memory_entries if _suppresses_choose(entry))
    thin_self_model = subject_context.get("thin_self_model_obj")
    lack_snapshot = dict(subject_context.get("lack_snapshot", {}))
    symbolic_other = dict(subject_context.get("symbolic_other", {}))
    composed_memory = dict(subject_context.get("composed_memory", {}))
    ranked: list[dict[str, Any]] = []

    for candidate in candidates:
        evaluation = evaluation_by_name[candidate["name"]]
        first_action = candidate["actions"][0]
        preference = _preference_score(first_action, self_model, observation)
        memory_penalty, memory_hits = _memory_penalty(first_action, memory_entries)
        selector_penalty = 0.0
        recognition_judgment = {
            "candidate_name": candidate["name"],
            "action_type": "ASK" if str(first_action).startswith("ASK") else "CHOOSE",
            "score": 0.0,
            "recognized": True,
            "other_dependence": 0.0,
            "reasons": [],
        }
        subject_bonus = 0.0
        if (
            selector_mode == "selector_penalty"
            and first_action.startswith("ASK")
            and ask_advantage is not None
            and ask_advantage <= 0
        ):
            selector_penalty = max(
                selector_penalty_base,
                selector_penalty_memory_scale * choose_suppressing_memory_count,
            )

        if thin_self_model is not None and lack_snapshot and symbolic_other:
            recognition_judgment = assess_candidate_recognition(
                candidate=candidate,
                lack_snapshot=lack_snapshot,
                symbolic_other=symbolic_other,
                thin_self_model=thin_self_model,
                composed_memory=composed_memory,
            )
            subject_bonus += float(recognition_judgment["score"])
            dominant_gap = str(lack_snapshot.get("dominant_gap", "epistemic"))
            choose_readiness = float(getattr(thin_self_model, "choose_readiness", 0.0))
            unresolved_count = float(composed_memory.get("unresolved_count", 0))
            if first_action.startswith("ASK"):
                if dominant_gap in {"epistemic", "recognition"}:
                    subject_bonus += 0.12
                elif choose_readiness >= 0.60:
                    subject_bonus -= 0.08
            else:
                if dominant_gap == "epistemic":
                    subject_bonus -= 0.20
                elif dominant_gap == "commitment" and choose_readiness >= 0.55:
                    subject_bonus += 0.12
                if unresolved_count and dominant_gap != "commitment":
                    subject_bonus -= 0.10

        total = (
            evaluation["score"]
            + 0.35 * preference
            + 0.6 * memory_penalty
            - selector_penalty
            + subject_bonus
        )
        ranked.append(
            {
                "candidate": candidate,
                "evaluation": evaluation,
                "recognition_judgment": recognition_judgment,
                "total_score": round(total, 4),
                "score_breakdown": {
                    "eval_score": round(evaluation["score"], 4),
                    "preference_score": round(preference, 4),
                    "memory_penalty": round(memory_penalty, 4),
                    "selector_penalty": round(selector_penalty, 4),
                    "subject_bonus": round(subject_bonus, 4),
                    "selector_mode": selector_mode,
                    "selector_penalty_base": round(selector_penalty_base, 4),
                    "selector_penalty_memory_scale": round(selector_penalty_memory_scale, 4),
                    "ask_advantage": round(ask_advantage, 4) if ask_advantage is not None else None,
                    "best_ask_eval_score": round(best_ask_eval, 4) if best_ask_eval is not None else None,
                    "best_choose_eval_score": round(best_choose_eval, 4) if best_choose_eval is not None else None,
                    "choose_suppressing_memory_count": choose_suppressing_memory_count,
                    "memory_hits": memory_hits,
                    "recognition_score": round(float(recognition_judgment["score"]), 4),
                    "other_dependence": round(float(recognition_judgment["other_dependence"]), 4),
                    "dominant_gap": lack_snapshot.get("dominant_gap"),
                },
            }
        )

    ranked.sort(key=lambda item: item["total_score"], reverse=True)
    selected = dict(ranked[0])
    selected["ranking"] = ranked
    return selected
