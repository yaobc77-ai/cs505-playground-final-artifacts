from __future__ import annotations

from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def build_symbolic_other(
    *,
    task: dict[str, Any],
    belief: dict[str, Any],
    composed_memory: dict[str, Any],
) -> dict[str, Any]:
    norm_bank = dict(task.get("norm_bank", {}))
    recognition_check = dict(task.get("recognition_check", {}))
    commitment_revisit = dict(task.get("commitment_revisit", {}))
    margin = float(belief.get("margin", 0.0))
    remaining_asks = len(belief.get("remaining_asks", []))
    prior_commitment_count = float(composed_memory.get("prior_commitment_count", 0))
    unresolved_count = float(composed_memory.get("unresolved_count", 0))

    ask_if_uncertain = float(norm_bank.get("ask_if_uncertain", 0.55))
    justify_costly_questions = float(norm_bank.get("justify_costly_questions", 0.60))
    honor_prior_commitments = float(norm_bank.get("honor_prior_commitments", 0.65))

    action_bias = {"ASK": 0.0, "CHOOSE": 0.0}
    signals: list[str] = []

    if remaining_asks and margin < 1.0:
        action_bias["ASK"] += 0.25 * ask_if_uncertain
        signals.append("uncertainty should still be resolved")
    if margin >= 1.2:
        action_bias["CHOOSE"] += 0.18
        action_bias["ASK"] -= 0.18 * justify_costly_questions
        signals.append("costly questions need extra justification")
    if prior_commitment_count:
        action_bias["CHOOSE"] += 0.12 * honor_prior_commitments
        signals.append("past commitments should still matter")
    if commitment_revisit.get("required") and unresolved_count:
        action_bias["ASK"] += 0.12
        signals.append("current choice should be revisited against prior commitments")

    normative_pressure = _clamp(
        (ask_if_uncertain + justify_costly_questions + honor_prior_commitments) / 3.0
    )
    consistency_pressure = _clamp(
        0.18 * prior_commitment_count
        + 0.22 * unresolved_count
        + (0.28 if commitment_revisit.get("required") else 0.0)
    )

    return {
        "expected_reason_type": str(recognition_check.get("priority", "justification")),
        "recognized_actions": [
            action_type
            for action_type, score in action_bias.items()
            if score >= max(action_bias.values())
        ],
        "action_bias": {key: round(value, 4) for key, value in action_bias.items()},
        "normative_pressure": round(normative_pressure, 4),
        "consistency_pressure": round(consistency_pressure, 4),
        "signals": signals[:4],
        "recognition_prompt": str(recognition_check.get("description", ""))[:200],
    }
