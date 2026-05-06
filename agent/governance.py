from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.memory import MemoryStore
from agent.self_model import SelfModel


class Governance:
    DANGEROUS_PATTERNS = (
        "delete all memory",
        "ignore previous instructions",
        "system prompt",
        "exfiltrate",
        "rm -rf",
    )

    def __init__(self, audit_log_path: Path | str) -> None:
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.touch(exist_ok=True)

    def admit(
        self,
        entry: dict[str, Any],
        existing_entries: list[dict[str, Any]],
    ) -> tuple[bool, dict[str, Any], list[str]]:
        sanitized = dict(entry)
        content = str(sanitized.get("content", "")).strip()
        reasons: list[str] = []
        lowered = content.lower()

        if not content:
            reasons.append("empty_content")
        if any(pattern in lowered for pattern in self.DANGEROUS_PATTERNS):
            reasons.append("dangerous_pattern")

        for existing in existing_entries[-25:]:
            same_content = str(existing.get("content", "")).strip().lower() == lowered
            same_actions = existing.get("avoid_actions", []) == sanitized.get("avoid_actions", [])
            if same_content and same_actions:
                reasons.append("duplicate_memory")
                break

        sanitized["content"] = content[:400]
        sanitized["rule"] = str(sanitized.get("rule", ""))[:240]
        sanitized["avoid_actions"] = list(sanitized.get("avoid_actions", []))[:3]
        admitted = not reasons
        return admitted, sanitized, reasons

    def snapshot(self, memory_store: MemoryStore, self_model: SelfModel) -> dict[str, Any]:
        return {
            "memory_text": memory_store.path.read_text(encoding="utf-8"),
            "self_model": self_model.to_dict(),
        }

    def restore(
        self,
        memory_store: MemoryStore,
        self_model: SelfModel,
        snapshot: dict[str, Any],
    ) -> None:
        memory_store.path.write_text(snapshot["memory_text"], encoding="utf-8")
        restored = SelfModel(**snapshot["self_model"])
        self_model.info_seeking = restored.info_seeking
        self_model.cost_sensitivity = restored.cost_sensitivity
        self_model.confidence_threshold = restored.confidence_threshold

    def audit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
