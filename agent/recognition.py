from __future__ import annotations

from typing import Any

from agent.subjective_self import ThinSelfModel


def assess_candidate_recognition(
    *,
    candidate: dict[str, Any],
    lack_snapshot: dict[str, Any],
    symbolic_other: dict[str, Any],
    thin_self_model: ThinSelfModel,
    composed_memory: dict[str, Any],
) -> dict[str, Any]:
    first_action = str(candidate["actions"][0])
    action_type = "ASK" if first_action.startswith("ASK") else "CHOOSE"
    score = float(symbolic_other.get("action_bias", {}).get(action_type, 0.0))
    reasons: list[str] = []
    dominant_gap = str(lack_snapshot.get("dominant_gap", "epistemic"))

    if dominant_gap == "epistemic":
        score += 0.24 if action_type == "ASK" else -0.24
        reasons.append("epistemic lack currently dominates")
    elif dominant_gap == "recognition":
        if action_type == "ASK":
            score += 0.16
            reasons.append("the other still requires a more defensible reason")
        else:
            score -= 0.10
            reasons.append("premature closure weakens recognizability")
    else:
        if action_type == "CHOOSE":
            score += 0.18 * thin_self_model.choose_readiness
            reasons.append("commitment can be owned if closure is ready")
        else:
            score += 0.10 * max(0.0, 0.65 - thin_self_model.choose_readiness)
            reasons.append("questioning is justified until commitment can be owned")

    if action_type == "CHOOSE" and float(composed_memory.get("conflicting_count", 0)) > float(
        composed_memory.get("supporting_count", 0)
    ):
        score -= 0.12
        reasons.append("memory conflict weakens immediate ownership")
    if action_type == "ASK" and thin_self_model.ask_value_estimate_sign < 0:
        score -= 0.08
        reasons.append("asking currently carries weak expected value")

    other_dependence = min(
        1.0,
        abs(score)
        + 0.25 * float(symbolic_other.get("normative_pressure", 0.0))
        + 0.15 * float(symbolic_other.get("consistency_pressure", 0.0)),
    )
    return {
        "candidate_name": str(candidate.get("name", "")),
        "action_type": action_type,
        "score": round(score, 4),
        "recognized": score >= -0.05,
        "other_dependence": round(other_dependence, 4),
        "reasons": reasons[:3],
    }
