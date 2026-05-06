from __future__ import annotations

from dataclasses import asdict, dataclass


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class SelfModel:
    info_seeking: float = 0.55
    cost_sensitivity: float = 0.45
    confidence_threshold: float = 1.25

    def update(self, delta: dict[str, float]) -> None:
        for key, value in delta.items():
            if not hasattr(self, key):
                continue
            current = getattr(self, key)
            if key == "confidence_threshold":
                setattr(self, key, _clamp(current + value, 0.4, 3.0))
            else:
                setattr(self, key, _clamp(current + value))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def clone(self) -> "SelfModel":
        return SelfModel(**self.to_dict())
