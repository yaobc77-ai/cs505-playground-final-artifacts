from __future__ import annotations

from typing import Any

from agent.subjective_self import ThinSelfModel


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def build_lack_snapshot(
    *,
    task: dict[str, Any],
    belief: dict[str, Any],
    thin_self_model: ThinSelfModel,
    symbolic_other: dict[str, Any],
    composed_memory: dict[str, Any],
) -> dict[str, Any]:
    margin = float(belief.get("margin", 0.0))
    remaining_asks = len(belief.get("remaining_asks", []))
    conflict_count = float(composed_memory.get("conflicting_count", 0))
    unresolved_count = float(composed_memory.get("unresolved_count", 0))
    prior_commitment_count = float(composed_memory.get("prior_commitment_count", 0))
    revisit_required = 1.0 if (task.get("commitment_revisit") or {}).get("required") else 0.0

    epistemic_gap = _clamp(
        ((1.15 - min(margin, 1.15)) / 1.15 if remaining_asks else 0.0)
        + 0.10 * max(0.0, 0.45 - thin_self_model.knowledge_sufficiency)
    )
    recognition_gap = _clamp(
        0.55 * float(symbolic_other.get("normative_pressure", 0.0))
        + 0.30 * float(symbolic_other.get("consistency_pressure", 0.0))
        + 0.10 * conflict_count
        - 0.18 * thin_self_model.memory_reliability
    )
    commitment_gap = _clamp(
        0.65 * (1.0 - thin_self_model.choose_readiness)
        + 0.10 * conflict_count
        + 0.10 * prior_commitment_count
        + 0.15 * unresolved_count
        + 0.18 * revisit_required
    )

    gap_values = {
        "epistemic": round(epistemic_gap, 4),
        "recognition": round(recognition_gap, 4),
        "commitment": round(commitment_gap, 4),
    }
    dominant_gap = max(gap_values, key=gap_values.get)

    return {
        **gap_values,
        "dominant_gap": dominant_gap,
        "needs_deferred_closure": dominant_gap in {"epistemic", "recognition"} or commitment_gap >= 0.6,
    }
