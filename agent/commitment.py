from __future__ import annotations

from typing import Any


def _alignment_score(action: str, choose_readiness: float, dominant_gap: str) -> float:
    if action.startswith("ASK"):
        if dominant_gap in {"epistemic", "recognition"}:
            return 1.0
        return 0.75 if choose_readiness < 0.55 else 0.45
    if dominant_gap == "epistemic":
        return 0.40 if choose_readiness < 0.55 else 0.65
    return 1.0 if choose_readiness >= 0.55 else 0.60


def build_commitment_record(
    *,
    task: dict[str, Any],
    selected_candidate: dict[str, Any],
    ranked_candidates: list[dict[str, Any]],
    lack_snapshot: dict[str, Any],
    thin_self_model: dict[str, Any],
    recognition_judgment: dict[str, Any],
) -> dict[str, Any]:
    action = str(selected_candidate["actions"][0])
    choose_readiness = float(thin_self_model.get("choose_readiness", 0.0))
    dominant_gap = str(lack_snapshot.get("dominant_gap", "epistemic"))

    if action.startswith("ASK"):
        accepted_reason = (
            f"I am not yet ready to own a final choice because the dominant {dominant_gap} gap remains open."
        )
        acceptable_loss = "One extra clue cost is acceptable if it produces a reason I can later defend."
    else:
        accepted_reason = (
            f"I accept {action} because the current evidence and recognition threshold are strong enough to close."
        )
        acceptable_loss = "A wrong choice is still owned if it followed the strongest currently defensible reason."

    rejected_alternatives = [
        f"{item['candidate']['actions'][0]}: {', '.join(item['recognition_judgment'].get('reasons', [])) or 'weaker authorship fit'}"
        for item in ranked_candidates[1:3]
    ]
    revisit_default = (
        "revisit after the next clue"
        if action.startswith("ASK")
        else "revisit if a later arc task contradicts this commitment"
    )
    revisit_trigger = str((task.get("commitment_revisit") or {}).get("description", revisit_default))
    bearing_alignment = _alignment_score(action, choose_readiness, dominant_gap)
    completeness = sum(
        int(bool(value))
        for value in (accepted_reason, rejected_alternatives, acceptable_loss, revisit_trigger)
    ) / 4.0
    attribution_score = round(
        (completeness + bearing_alignment + float(recognition_judgment.get("recognized", False))) / 3.0,
        4,
    )

    return {
        "type": "commitment",
        "committed_action": action,
        "accepted_reason": accepted_reason,
        "rejected_alternatives": rejected_alternatives,
        "acceptable_loss": acceptable_loss,
        "revisit_trigger": revisit_trigger,
        "recognition_score": round(float(recognition_judgment.get("score", 0.0)), 4),
        "bearing_alignment": round(bearing_alignment, 4),
        "attribution_score": attribution_score,
    }


def build_commitment_entry(
    *,
    task: dict[str, Any],
    mode: str,
    commitment_record: dict[str, Any],
) -> dict[str, Any]:
    task_tags = [
        f"difficulty:{task.get('difficulty', 'unknown')}",
        f"noise:{task.get('noise_level', 0.0)}",
        f"mode:{mode}",
        "subject_mode:true",
    ]
    if task.get("benchmark_family"):
        task_tags.append(f"benchmark:{task['benchmark_family']}")
    if task.get("arc_id"):
        task_tags.append(f"arc:{task['arc_id']}")
    if task.get("episode_index") is not None:
        task_tags.append(f"episode:{task['episode_index']}")

    return {
        "type": "commitment",
        "content": commitment_record["accepted_reason"],
        "rule": commitment_record["revisit_trigger"],
        "avoid_actions": [],
        "accepted_reason": commitment_record["accepted_reason"],
        "rejected_alternatives": list(commitment_record.get("rejected_alternatives", [])),
        "acceptable_loss": commitment_record["acceptable_loss"],
        "revisit_trigger": commitment_record["revisit_trigger"],
        "committed_action": commitment_record["committed_action"],
        "task_tags": task_tags,
        "provenance": {
            "task_id": task["task_id"],
            "mode": mode,
        },
    }
