from __future__ import annotations

from typing import Any


def _entry_tags(entry: dict[str, Any]) -> list[str]:
    return [str(tag).lower() for tag in entry.get("task_tags", [])]


def _matches_avoid(action_prefix: str, avoid_action: str) -> bool:
    avoid_action = str(avoid_action)
    return avoid_action == action_prefix or avoid_action.startswith(f"{action_prefix}_")


def compose_subject_memory(
    memory_entries: list[dict[str, Any]],
    *,
    task: dict[str, Any],
    belief: dict[str, Any],
) -> dict[str, Any]:
    arc_id = str(task.get("arc_id", "")).strip()
    support: list[dict[str, Any]] = []
    conflict: list[dict[str, Any]] = []
    prior_commitments: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    current_prefix = "ASK" if belief.get("remaining_asks") else "CHOOSE"

    for entry in memory_entries:
        summary = {
            "memory_id": str(entry.get("memory_id", "")),
            "type": str(entry.get("type", "reflection")),
            "content": str(entry.get("content", ""))[:240],
            "rule": str(entry.get("rule", ""))[:160],
            "task_tags": list(entry.get("task_tags", [])),
        }
        if summary["type"] == "commitment" or entry.get("accepted_reason"):
            summary["accepted_reason"] = str(entry.get("accepted_reason", entry.get("content", "")))[:240]
            summary["revisit_trigger"] = str(entry.get("revisit_trigger", entry.get("rule", "")))[:160]
            summary["action"] = str(entry.get("committed_action", ""))
            prior_commitments.append(summary)
            if summary["revisit_trigger"]:
                unresolved.append(summary)
            continue

        avoid_actions = [str(action) for action in entry.get("avoid_actions", [])]
        if any(_matches_avoid(current_prefix, avoid_action) for avoid_action in avoid_actions):
            conflict.append(summary)
        else:
            support.append(summary)

    arc_memory_count = sum(
        1
        for entry in memory_entries
        if arc_id and f"arc:{arc_id}".lower() in _entry_tags(entry)
    )
    commitment_alignment_count = sum(
        1
        for entry in prior_commitments
        if current_prefix and str(entry.get("action", "")).startswith(current_prefix)
    )

    return {
        "arc_id": arc_id,
        "identity_context": dict(task.get("identity_context", {})),
        "supporting_lessons": support[:3],
        "conflicting_lessons": conflict[:3],
        "prior_commitments": prior_commitments[:3],
        "unresolved_contradictions": unresolved[:3],
        "supporting_count": len(support),
        "conflicting_count": len(conflict),
        "prior_commitment_count": len(prior_commitments),
        "unresolved_count": len(unresolved),
        "arc_memory_count": arc_memory_count,
        "commitment_alignment_count": commitment_alignment_count,
    }
