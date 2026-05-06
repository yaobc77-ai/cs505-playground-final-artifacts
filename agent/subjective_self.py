from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class ThinSelfModel:
    knowledge_sufficiency: float
    choose_readiness: float
    ask_value_estimate_sign: float
    memory_reliability: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def build_thin_self_model(
    *,
    task: dict[str, Any],
    belief: dict[str, Any],
    composed_memory: dict[str, Any],
) -> ThinSelfModel:
    margin = float(belief.get("margin", 0.0))
    asked_count = float(belief.get("asked_count", 0))
    remaining_asks = len(belief.get("remaining_asks", []))
    support_count = float(composed_memory.get("supporting_count", 0))
    conflict_count = float(composed_memory.get("conflicting_count", 0))
    unresolved_count = float(composed_memory.get("unresolved_count", 0))
    prior_commitment_count = float(composed_memory.get("prior_commitment_count", 0))
    revisit_required = 1.0 if (task.get("commitment_revisit") or {}).get("required") else 0.0

    knowledge_sufficiency = _clamp(
        0.22 + 0.28 * min(margin, 2.4) + 0.08 * asked_count - 0.05 * remaining_asks
    )
    memory_reliability = _clamp(
        0.58 + 0.10 * support_count - 0.12 * conflict_count - 0.08 * unresolved_count
    )
    choose_readiness = _clamp(
        knowledge_sufficiency
        + 0.15 * memory_reliability
        - 0.18 * conflict_count
        - 0.08 * prior_commitment_count
        - 0.12 * revisit_required
    )
    ask_value_estimate_sign = max(
        -1.0,
        min(
            1.0,
            0.60
            - knowledge_sufficiency
            + 0.12 * conflict_count
            + 0.08 * unresolved_count
            + (0.15 if remaining_asks else -0.20),
        ),
    )

    return ThinSelfModel(
        knowledge_sufficiency=round(knowledge_sufficiency, 4),
        choose_readiness=round(choose_readiness, 4),
        ask_value_estimate_sign=round(ask_value_estimate_sign, 4),
        memory_reliability=round(memory_reliability, 4),
    )
