from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent.self_model import SelfModel

CHOICES = ("A", "B", "C")


def _tool_family(action: str) -> str:
    if action.startswith("ASK_SCAN_"):
        return "scan"
    if action.startswith("ASK_VERIFY_"):
        return "verify"
    if action.startswith("ASK_CLUE_"):
        return "clue"
    return "other"


def _action_sort_key(action: str) -> tuple[int, str]:
    try:
        suffix = int(str(action).rsplit("_", 1)[-1])
    except ValueError:
        suffix = 10**9
    return suffix, str(action)


def belief_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    scores = {choice: 0.0 for choice in CHOICES}
    asked_clues = list(observation.get("asked_clues", []))
    clue_top_choices: list[str] = []
    reliabilities: list[float] = []
    for clue in asked_clues:
        reliability = float(clue.get("reliability", 1.0))
        reliabilities.append(reliability)
        clue_scores = dict(clue.get("scores", {}))
        weighted_scores = {
            choice: float(clue_scores.get(choice, 0.0)) * reliability
            for choice in CHOICES
        }
        for choice, value in clue.get("scores", {}).items():
            scores[choice] += float(value) * reliability
        ranked_clue = sorted(weighted_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        clue_top_choices.append(ranked_clue[0][0])

    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_choice, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else top_score
    remaining_asks = [
        action
        for action in observation.get("options", [])
        if action.startswith("ASK")
    ]
    remaining_asks.sort(key=_action_sort_key)
    evidence_count = len(asked_clues)
    mean_reliability = sum(reliabilities) / evidence_count if reliabilities else 0.0
    evidence_conflict = (
        sum(clue_top != top_choice for clue_top in clue_top_choices) / evidence_count
        if evidence_count
        else 0.0
    )
    return {
        "scores": scores,
        "ranked_choices": [choice for choice, _ in ranked],
        "top_choice": top_choice,
        "top_score": top_score,
        "margin": top_score - second_score,
        "remaining_asks": remaining_asks,
        "remaining_scan_actions": [action for action in remaining_asks if _tool_family(action) == "scan"],
        "remaining_verify_actions": [action for action in remaining_asks if _tool_family(action) == "verify"],
        "evidence_count": evidence_count,
        "mean_reliability": round(mean_reliability, 4),
        "evidence_conflict": round(evidence_conflict, 4),
        "asked_count": evidence_count,
    }


def generate_candidates(
    observation: dict[str, Any],
    self_model: SelfModel,
    candidate_limit: int = 5,
) -> list[dict[str, Any]]:
    belief = belief_from_observation(observation)
    remaining_asks = list(belief["remaining_asks"])
    scan_actions = list(belief.get("remaining_scan_actions", []))
    verify_actions = list(belief.get("remaining_verify_actions", []))
    legacy_actions = [
        action
        for action in remaining_asks
        if _tool_family(action) == "clue"
    ]
    ranked_choices = belief["ranked_choices"]
    top_choice = ranked_choices[0]
    second_choice = ranked_choices[1]
    candidates: list[dict[str, Any]] = []

    candidates.extend(
        [
            {
                "name": f"choose_{top_choice.lower()}",
                "actions": [f"CHOOSE_{top_choice}"],
                "category": "choose_now",
            },
            {
                "name": f"choose_{second_choice.lower()}",
                "actions": [f"CHOOSE_{second_choice}"],
                "category": "choose_alt",
            },
        ]
    )

    if scan_actions or verify_actions:
        if scan_actions:
            candidates.append(
                {
                    "name": f"{scan_actions[0].lower()}_then_choose_best",
                    "actions": [scan_actions[0], "CHOOSE_BEST"],
                    "category": "ask_then_choose",
                }
            )
        if verify_actions:
            candidates.append(
                {
                    "name": f"{verify_actions[0].lower()}_then_choose_best",
                    "actions": [verify_actions[0], "CHOOSE_BEST"],
                    "category": "ask_then_choose",
                }
            )

        preferred_tool_action: str | None = None
        if verify_actions and (
            belief["margin"] < self_model.confidence_threshold
            or float(belief.get("mean_reliability", 0.0)) < 0.8
            or float(belief.get("evidence_conflict", 0.0)) >= 0.34
        ):
            preferred_tool_action = verify_actions[0]
        elif scan_actions:
            preferred_tool_action = scan_actions[0]
        elif verify_actions:
            preferred_tool_action = verify_actions[0]

        if preferred_tool_action and self_model.info_seeking >= 0.6:
            candidates.append(
                {
                    "name": f"{preferred_tool_action.lower()}_only",
                    "actions": [preferred_tool_action],
                    "category": "ask_only",
                }
            )
    else:
        for ask_action in legacy_actions[:2]:
            candidates.append(
                {
                    "name": f"{ask_action.lower()}_then_choose_best",
                    "actions": [ask_action, "CHOOSE_BEST"],
                    "category": "ask_then_choose",
                }
            )

        if remaining_asks and self_model.info_seeking >= 0.6:
            candidates.append(
                {
                    "name": f"{remaining_asks[0].lower()}_only",
                    "actions": [remaining_asks[0]],
                    "category": "ask_only",
                }
            )

        if remaining_asks:
            candidates.append(
                {
                    "name": f"{remaining_asks[0].lower()}_then_choose_{top_choice.lower()}",
                    "actions": [remaining_asks[0], f"CHOOSE_{top_choice}"],
                    "category": "ask_then_choose_fixed",
                }
            )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate["actions"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= candidate_limit:
            break
    return unique


def resolve_action(action: str, observation: dict[str, Any]) -> str:
    if action != "CHOOSE_BEST":
        return action
    belief = belief_from_observation(observation)
    return f"CHOOSE_{belief['top_choice']}"


def evaluate_candidate(env: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    sim_env = env.clone()
    observation = deepcopy(sim_env.get_observation())
    total_reward = 0.0
    realized_actions: list[str] = []
    done = False

    for action in candidate["actions"]:
        resolved = resolve_action(action, observation)
        observation, reward, done, _ = sim_env.step(resolved)
        total_reward += reward
        realized_actions.append(resolved)
        if done:
            break

    return {
        "candidate_name": candidate["name"],
        "score": round(total_reward, 4),
        "done": done,
        "realized_actions": realized_actions,
    }
