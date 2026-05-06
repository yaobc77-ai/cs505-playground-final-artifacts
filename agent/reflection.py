from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


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


def build_reflection(
    *,
    task: dict[str, Any],
    mode: str,
    step_records: list[dict[str, Any]],
    total_reward: float,
    success: bool,
) -> dict[str, Any]:
    asked_actions = [record for record in step_records if record["action"].startswith("ASK")]
    last_step = step_records[-1]
    belief = last_step["belief"]
    margin = float(belief["margin"])

    rule = ""
    avoid_actions: list[str] = []
    z_delta: dict[str, float] = {}

    if not success and last_step["action"].startswith("CHOOSE") and belief["remaining_asks"]:
        rule = "When clues remain and the belief margin is small, ask one more clue before choosing."
        avoid_actions = ["CHOOSE"]
        z_delta = {"info_seeking": 0.12, "cost_sensitivity": -0.04}
    elif success and len(asked_actions) >= 2 and total_reward < 0.95:
        rule = "Once one answer has a clear lead, avoid extra clue requests that only add cost."
        avoid_actions = ["ASK"]
        z_delta = {"info_seeking": -0.05, "cost_sensitivity": 0.08}
    elif success and len(asked_actions) == 0 and margin >= 1.5:
        rule = "Direct choice worked because the current evidence gap was already wide."
        z_delta = {"cost_sensitivity": 0.04, "confidence_threshold": -0.05}
    else:
        rule = "Use the current belief margin to decide whether to ask again or commit."
        z_delta = {"confidence_threshold": 0.02 if not success else -0.02}

    return {
        "type": "reflection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content": rule,
        "rule": rule,
        "avoid_actions": avoid_actions,
        "z_delta": z_delta,
        "task_tags": _task_tags(task, mode),
        "provenance": {
            "task_id": task["task_id"],
            "mode": mode,
        },
    }
