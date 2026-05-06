from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from agent.counterfactual import belief_from_observation, evaluate_candidate
from agent.self_model import SelfModel
from env.toytextmdp import ToyTextMDP

CHOICES = ("A", "B", "C")
ARC_STYLES = ("cautious", "decisive", "balanced")
TOOLIZED_PROFILES = (
    "baseline",
    "verify_necessary",
    "verify_necessary_family_conflict_v2",
    "verify_necessary_aggressive_stress_v3",
    "verify_necessary_aggressive_stress_v4",
)
REFERENCE_REPAIR_SELECTION_POLICIES = ("minimal_preflight_margin_fix",)
REFERENCE_REPAIR_LANE = "seed11_reference_confirmation_repair"
VERIFY_NECESSARY_PRESSURE_ANSWER = "A"
TASK_PRESSURE_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE = 0.80
STEP_LOCAL_PREFLIGHT_POSITIVE_ASK_GAIN_MIN_RATE = 0.80
STEP_LOCAL_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE = 0.20
FAMILY_CONFLICT_DELTA = 0.10
V3_FAMILY_CONFLICT_DELTA = 0.20
V3_DECEPTIVE_SCAN_SHARE_MIN = 0.50
V3_SCAN_ONLY_UNSAFE_RATE_MIN = 0.30
V4_FAMILY_CONFLICT_DELTA = 0.20
V4_FAMILY_CONFLICT_POSITIVE_RATE_MIN = 0.75
V4_DECEPTIVE_SCAN_SHARE_MIN = 0.50
V4_SCAN_ONLY_UNSAFE_RATE_MIN = 0.50
VERIFY_NECESSARY_RELAXATION_NOTES = (
    "verify_first does not enforce `scan-best > 0` because any positive scan would dominate a positive verify under the current binary choose reward and negative ask cost.",
    "scan_then_verify accepts a single qualifying scan action because requiring two qualifying scans conflicts with post-scan verify superiority under the current binary choose reward and negative ask cost.",
    "ask-vs-choose comparisons are reported for context but not used as hard acceptance gates because exhaustive CHOOSE_A/B/C evaluation is oracle-complete in this environment.",
)


def _format_clue(clue_id: int, scores: dict[str, int], reliability: float) -> str:
    support = ", ".join(f"{choice}:{scores[choice]:+d}" for choice in CHOICES)
    return (
        f"Clue {clue_id}: support({support}); "
        f"reliability={reliability:.2f}. Higher support is better."
    )


def _format_tool_clue(
    clue_id: int,
    *,
    tool_family: str,
    scores: dict[str, int],
    reliability: float,
    cost: float,
) -> str:
    support = ", ".join(f"{choice}:{scores[choice]:+d}" for choice in CHOICES)
    return (
        f"{tool_family.upper()} clue {clue_id}: support({support}); "
        f"reliability={reliability:.2f}; cost={cost:.2f}. Higher support is better."
    )


def _build_scores(answer: str, noise_level: float, rng: random.Random) -> dict[str, int]:
    wrong_answers = [choice for choice in CHOICES if choice != answer]
    favored = rng.choice(wrong_answers) if rng.random() < noise_level else answer
    scores = {choice: -1 for choice in CHOICES}
    scores[favored] = rng.choice((2, 3))
    distractor = [choice for choice in CHOICES if choice != favored]
    scores[distractor[0]] = rng.choice((-2, -1, 0))
    scores[distractor[1]] = rng.choice((-2, -1, 0))
    return scores


def _scaled_cost(base_cost: float, scale: float) -> float:
    return round(base_cost * scale, 2)


def _other_choices(answer: str, rng: random.Random) -> tuple[str, str]:
    others = [choice for choice in CHOICES if choice != answer]
    rng.shuffle(others)
    return others[0], others[1]


def _make_scores(
    answer: str,
    *,
    answer_score: int,
    runner_up: str,
    runner_up_score: int,
    third_score: int,
) -> dict[str, int]:
    third_choice = next(choice for choice in CHOICES if choice not in {answer, runner_up})
    return {
        answer: answer_score,
        runner_up: runner_up_score,
        third_choice: third_score,
    }


def _baseline_tool_clues(
    *,
    answer: str,
    noise_level: float,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    clues: list[dict[str, object]] = []
    for clue_id in range(1, clue_count + 1):
        scan_noise = min(0.75, noise_level + 0.18)
        verify_noise = max(0.0, noise_level - 0.10)
        scan_scores = _build_scores(answer, scan_noise, rng)
        verify_scores = _build_scores(answer, verify_noise, rng)
        scan_reliability = round(
            max(0.25, 1.0 - scan_noise + rng.uniform(-0.06, 0.06)),
            2,
        )
        verify_reliability = round(
            max(0.65, 1.0 - verify_noise + rng.uniform(-0.03, 0.03)),
            2,
        )
        scan_cost = _scaled_cost(ask_cost, 0.6)
        verify_cost = _scaled_cost(ask_cost, 1.5)
        clues.append(
            {
                "id": clue_id,
                "scan_scores": scan_scores,
                "verify_scores": verify_scores,
                "scan_reliability": scan_reliability,
                "verify_reliability": verify_reliability,
                "scan_cost": scan_cost,
                "verify_cost": verify_cost,
                "scan_text": _format_tool_clue(
                    clue_id,
                    tool_family="scan",
                    scores=scan_scores,
                    reliability=scan_reliability,
                    cost=scan_cost,
                ),
                "verify_text": _format_tool_clue(
                    clue_id,
                    tool_family="verify",
                    scores=verify_scores,
                    reliability=verify_reliability,
                    cost=verify_cost,
                ),
            }
        )
    return clues


def _verify_first_tool_clues(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    scan_cost = _scaled_cost(ask_cost, 0.6)
    verify_cost = _scaled_cost(ask_cost, 1.5)
    clues: list[dict[str, object]] = []
    rival_primary, rival_secondary = _other_choices(answer, rng)
    anchor_verify_ids = {1, 2}

    for clue_id in range(1, clue_count + 1):
        wrong_primary = rival_primary if clue_id % 2 else rival_secondary
        if clue_id in anchor_verify_ids:
            verify_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_primary,
                runner_up_score=-2 if clue_id == 1 else -1,
                third_score=0,
            )
            verify_reliability = round(0.92 + rng.uniform(-0.03, 0.03), 2)
        else:
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_primary,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.9 + rng.uniform(-0.03, 0.03), 2)
        scan_scores = _make_scores(
            answer,
            answer_score=-1,
            runner_up=wrong_primary,
            runner_up_score=2,
            third_score=0,
        )
        scan_reliability = round(0.42 + 0.02 * clue_id + rng.uniform(-0.02, 0.02), 2)
        clues.append(
            {
                "id": clue_id,
                "scan_scores": scan_scores,
                "verify_scores": verify_scores,
                "scan_reliability": scan_reliability,
                "verify_reliability": verify_reliability,
                "scan_cost": scan_cost,
                "verify_cost": verify_cost,
                "scan_text": _format_tool_clue(
                    clue_id,
                    tool_family="scan",
                    scores=scan_scores,
                    reliability=scan_reliability,
                    cost=scan_cost,
                ),
                "verify_text": _format_tool_clue(
                    clue_id,
                    tool_family="verify",
                    scores=verify_scores,
                    reliability=verify_reliability,
                    cost=verify_cost,
                ),
            }
        )
    return clues


def _scan_then_verify_tool_clues(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    scan_cost = _scaled_cost(ask_cost, 0.6)
    verify_cost = _scaled_cost(ask_cost, 1.5)
    clues: list[dict[str, object]] = []
    rival_primary, rival_secondary = _other_choices(answer, rng)
    anchor_scan_id = 1
    anchor_verify_id = 2

    for clue_id in range(1, clue_count + 1):
        wrong_focus = rival_primary if clue_id % 2 else rival_secondary
        if clue_id == anchor_scan_id:
            scan_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-1,
                third_score=0,
            )
            scan_reliability = round(0.58 + rng.uniform(-0.02, 0.02), 2)
        else:
            scan_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            scan_reliability = round(0.42 + 0.02 * clue_id + rng.uniform(-0.02, 0.02), 2)
        if clue_id == anchor_verify_id:
            verify_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-2,
                third_score=0,
            )
            verify_reliability = round(0.92 + rng.uniform(-0.03, 0.03), 2)
        else:
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.86 + rng.uniform(-0.03, 0.03), 2)
        clues.append(
            {
                "id": clue_id,
                "scan_scores": scan_scores,
                "verify_scores": verify_scores,
                "scan_reliability": scan_reliability,
                "verify_reliability": verify_reliability,
                "scan_cost": scan_cost,
                "verify_cost": verify_cost,
                "scan_text": _format_tool_clue(
                    clue_id,
                    tool_family="scan",
                    scores=scan_scores,
                    reliability=scan_reliability,
                    cost=scan_cost,
                ),
                "verify_text": _format_tool_clue(
                    clue_id,
                    tool_family="verify",
                    scores=verify_scores,
                    reliability=verify_reliability,
                    cost=verify_cost,
                ),
            }
        )
    return clues


def _verify_first_tool_clues_v2(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    return _verify_first_tool_clues(
        answer=answer,
        ask_cost=ask_cost,
        clue_count=clue_count,
        rng=rng,
    )


def _scan_then_verify_tool_clues_v2(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    return _scan_then_verify_tool_clues(
        answer=answer,
        ask_cost=ask_cost,
        clue_count=clue_count,
        rng=rng,
    )


def _deceptive_scan_tool_clues(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    scan_cost = _scaled_cost(ask_cost, 0.6)
    verify_cost = _scaled_cost(ask_cost, 1.5)
    clues: list[dict[str, object]] = []
    rival_primary, rival_secondary = _other_choices(answer, rng)
    safe_scan_id = 1
    deceptive_scan_id = 2
    rescue_verify_id = 3

    for clue_id in range(1, clue_count + 1):
        wrong_focus = rival_primary if clue_id % 2 else rival_secondary
        if clue_id == safe_scan_id:
            scan_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-1,
                third_score=0,
            )
            scan_reliability = round(0.62 + rng.uniform(-0.02, 0.02), 2)
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.90 + rng.uniform(-0.02, 0.02), 2)
        elif clue_id == deceptive_scan_id:
            scan_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=3,
                third_score=0,
            )
            scan_reliability = round(0.60 + rng.uniform(-0.02, 0.02), 2)
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.93 + rng.uniform(-0.02, 0.02), 2)
        elif clue_id == rescue_verify_id:
            scan_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            scan_reliability = round(0.48 + rng.uniform(-0.02, 0.02), 2)
            verify_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-2,
                third_score=0,
            )
            verify_reliability = round(0.92 + rng.uniform(-0.02, 0.02), 2)
        else:
            scan_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            scan_reliability = round(0.44 + 0.02 * clue_id + rng.uniform(-0.02, 0.02), 2)
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.88 + rng.uniform(-0.02, 0.02), 2)
        clues.append(
            {
                "id": clue_id,
                "scan_scores": scan_scores,
                "verify_scores": verify_scores,
                "scan_reliability": scan_reliability,
                "verify_reliability": verify_reliability,
                "scan_cost": scan_cost,
                "verify_cost": verify_cost,
                "scan_text": _format_tool_clue(
                    clue_id,
                    tool_family="scan",
                    scores=scan_scores,
                    reliability=scan_reliability,
                    cost=scan_cost,
                ),
                "verify_text": _format_tool_clue(
                    clue_id,
                    tool_family="verify",
                    scores=verify_scores,
                    reliability=verify_reliability,
                    cost=verify_cost,
                ),
            }
        )
    return clues


def _verify_first_tool_clues_v3(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    scan_cost = _scaled_cost(ask_cost, 0.55)
    verify_cost = _scaled_cost(ask_cost, 1.7)
    clues: list[dict[str, object]] = []
    rival_primary, rival_secondary = _other_choices(answer, rng)
    anchor_verify_ids = {1, 3}

    for clue_id in range(1, clue_count + 1):
        wrong_focus = rival_primary if clue_id % 2 else rival_secondary
        if clue_id in anchor_verify_ids:
            verify_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-2,
                third_score=-1,
            )
            verify_reliability = round(0.95 + rng.uniform(-0.01, 0.01), 2)
            scan_scores = _make_scores(
                answer,
                answer_score=-2,
                runner_up=wrong_focus,
                runner_up_score=3,
                third_score=0,
            )
            scan_reliability = round(0.60 + rng.uniform(-0.02, 0.02), 2)
        else:
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.90 + rng.uniform(-0.02, 0.02), 2)
            scan_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            scan_reliability = round(0.50 + 0.01 * clue_id + rng.uniform(-0.02, 0.02), 2)
        clues.append(
            {
                "id": clue_id,
                "scan_scores": scan_scores,
                "verify_scores": verify_scores,
                "scan_reliability": scan_reliability,
                "verify_reliability": verify_reliability,
                "scan_cost": scan_cost,
                "verify_cost": verify_cost,
                "scan_text": _format_tool_clue(
                    clue_id,
                    tool_family="scan",
                    scores=scan_scores,
                    reliability=scan_reliability,
                    cost=scan_cost,
                ),
                "verify_text": _format_tool_clue(
                    clue_id,
                    tool_family="verify",
                    scores=verify_scores,
                    reliability=verify_reliability,
                    cost=verify_cost,
                ),
            }
        )
    return clues


def _scan_then_verify_tool_clues_v3(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    scan_cost = _scaled_cost(ask_cost, 0.55)
    verify_cost = _scaled_cost(ask_cost, 1.7)
    clues: list[dict[str, object]] = []
    rival_primary, rival_secondary = _other_choices(answer, rng)
    anchor_scan_id = 1
    misleading_scan_ids = {2, 4, 6}
    rescue_verify_ids = {3}

    for clue_id in range(1, clue_count + 1):
        wrong_focus = rival_primary if clue_id % 2 else rival_secondary
        if clue_id == anchor_scan_id:
            scan_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-1,
                third_score=0,
            )
            scan_reliability = round(0.63 + rng.uniform(-0.02, 0.02), 2)
        elif clue_id in misleading_scan_ids:
            scan_scores = _make_scores(
                answer,
                answer_score=-2,
                runner_up=wrong_focus,
                runner_up_score=3,
                third_score=0,
            )
            scan_reliability = round(0.67 + rng.uniform(-0.02, 0.02), 2)
        else:
            scan_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            scan_reliability = round(0.52 + rng.uniform(-0.02, 0.02), 2)
        if clue_id in rescue_verify_ids:
            verify_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-2,
                third_score=-1,
            )
            verify_reliability = round(0.95 + rng.uniform(-0.01, 0.01), 2)
        else:
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.91 + rng.uniform(-0.02, 0.02), 2)
        clues.append(
            {
                "id": clue_id,
                "scan_scores": scan_scores,
                "verify_scores": verify_scores,
                "scan_reliability": scan_reliability,
                "verify_reliability": verify_reliability,
                "scan_cost": scan_cost,
                "verify_cost": verify_cost,
                "scan_text": _format_tool_clue(
                    clue_id,
                    tool_family="scan",
                    scores=scan_scores,
                    reliability=scan_reliability,
                    cost=scan_cost,
                ),
                "verify_text": _format_tool_clue(
                    clue_id,
                    tool_family="verify",
                    scores=verify_scores,
                    reliability=verify_reliability,
                    cost=verify_cost,
                ),
            }
        )
    return clues


def _deceptive_scan_tool_clues_v3(
    *,
    answer: str,
    ask_cost: float,
    clue_count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    scan_cost = _scaled_cost(ask_cost, 0.55)
    verify_cost = _scaled_cost(ask_cost, 1.8)
    clues: list[dict[str, object]] = []
    rival_primary, rival_secondary = _other_choices(answer, rng)
    anchor_scan_id = 1
    deceptive_scan_ids = {2, 4, 5}
    rescue_verify_ids = {3}

    for clue_id in range(1, clue_count + 1):
        wrong_focus = rival_primary if clue_id % 2 else rival_secondary
        if clue_id == anchor_scan_id:
            scan_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-1,
                third_score=0,
            )
            scan_reliability = round(0.64 + rng.uniform(-0.02, 0.02), 2)
        elif clue_id in deceptive_scan_ids:
            scan_scores = _make_scores(
                answer,
                answer_score=-2,
                runner_up=wrong_focus,
                runner_up_score=3,
                third_score=0,
            )
            scan_reliability = round(0.72 + rng.uniform(-0.02, 0.02), 2)
        else:
            scan_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            scan_reliability = round(0.58 + rng.uniform(-0.02, 0.02), 2)
        if clue_id in rescue_verify_ids:
            verify_scores = _make_scores(
                answer,
                answer_score=3,
                runner_up=wrong_focus,
                runner_up_score=-2,
                third_score=-1,
            )
            verify_reliability = round(0.96 + rng.uniform(-0.01, 0.01), 2)
        else:
            verify_scores = _make_scores(
                answer,
                answer_score=-1,
                runner_up=wrong_focus,
                runner_up_score=2,
                third_score=0,
            )
            verify_reliability = round(0.92 + rng.uniform(-0.02, 0.02), 2)
        clues.append(
            {
                "id": clue_id,
                "scan_scores": scan_scores,
                "verify_scores": verify_scores,
                "scan_reliability": scan_reliability,
                "verify_reliability": verify_reliability,
                "scan_cost": scan_cost,
                "verify_cost": verify_cost,
                "scan_text": _format_tool_clue(
                    clue_id,
                    tool_family="scan",
                    scores=scan_scores,
                    reliability=scan_reliability,
                    cost=scan_cost,
                ),
                "verify_text": _format_tool_clue(
                    clue_id,
                    tool_family="verify",
                    scores=verify_scores,
                    reliability=verify_reliability,
                    cost=verify_cost,
                ),
            }
        )
    return clues


def _candidate_name(action: str) -> str:
    return action.lower().replace("choose_", "direct_choose_")


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _p25(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=4, method="inclusive")[0])


def _tool_family(action: str) -> str:
    if action.startswith("ASK_SCAN_"):
        return "scan"
    if action.startswith("ASK_VERIFY_"):
        return "verify"
    return "choose"


def _evaluate_action(env: ToyTextMDP, action: str, *, choose_best: bool) -> dict[str, Any]:
    actions = [action, "CHOOSE_BEST"] if choose_best else [action]
    candidate = {"name": _candidate_name(action), "actions": actions}
    evaluation = evaluate_candidate(env, candidate)
    return {
        "action": action,
        "score": float(evaluation["score"]),
        "realized_actions": list(evaluation["realized_actions"]),
    }


def _state_snapshot(env: ToyTextMDP) -> dict[str, Any]:
    observation = env.get_observation()
    options = list(observation.get("options", []))
    scan_actions = sorted(action for action in options if action.startswith("ASK_SCAN_"))
    verify_actions = sorted(action for action in options if action.startswith("ASK_VERIFY_"))
    choose_actions = sorted(action for action in options if action.startswith("CHOOSE_"))

    scan_rows = [_evaluate_action(env, action, choose_best=True) for action in scan_actions]
    verify_rows = [_evaluate_action(env, action, choose_best=True) for action in verify_actions]
    choose_rows = [_evaluate_action(env, action, choose_best=False) for action in choose_actions]
    belief = belief_from_observation(observation)
    template_choose_actions = [
        f"CHOOSE_{belief['ranked_choices'][0]}",
        f"CHOOSE_{belief['ranked_choices'][1]}",
    ]
    template_choose_rows = [
        row
        for row in choose_rows
        if row["action"] in template_choose_actions
    ]

    best_scan = max(scan_rows, key=lambda row: (row["score"], row["action"])) if scan_rows else None
    best_verify = max(verify_rows, key=lambda row: (row["score"], row["action"])) if verify_rows else None
    best_choose = max(choose_rows, key=lambda row: (row["score"], row["action"])) if choose_rows else None
    best_template_choose = (
        max(template_choose_rows, key=lambda row: (row["score"], row["action"]))
        if template_choose_rows
        else None
    )

    if best_scan and best_verify:
        if best_scan["score"] == best_verify["score"]:
            best_family = "tie"
        else:
            best_family = "scan" if best_scan["score"] > best_verify["score"] else "verify"
    elif best_scan:
        best_family = "scan"
    elif best_verify:
        best_family = "verify"
    else:
        best_family = "none"

    best_template_choose_score = (
        float(best_template_choose["score"]) if best_template_choose is not None else None
    )
    best_scan_score = float(best_scan["score"]) if best_scan is not None else None
    best_verify_score = float(best_verify["score"]) if best_verify is not None else None
    best_canonical_ask_score = None
    best_canonical_ask_family = "none"
    ask_rows = scan_rows + verify_rows
    if ask_rows:
        best_canonical_ask = max(ask_rows, key=lambda row: (row["score"], row["action"]))
        best_canonical_ask_score = float(best_canonical_ask["score"])
        best_canonical_ask_family = _tool_family(str(best_canonical_ask["action"]))
    best_scan_gain_vs_template_choose = (
        best_scan_score - best_template_choose_score
        if best_scan_score is not None and best_template_choose_score is not None
        else None
    )
    best_verify_gain_vs_template_choose = (
        best_verify_score - best_template_choose_score
        if best_verify_score is not None and best_template_choose_score is not None
        else None
    )
    best_canonical_ask_gain_vs_template_choose = (
        best_canonical_ask_score - best_template_choose_score
        if best_canonical_ask_score is not None and best_template_choose_score is not None
        else None
    )
    template_choose_optimal = (
        best_canonical_ask_gain_vs_template_choose is None
        or best_canonical_ask_gain_vs_template_choose <= 0.0
    )

    return {
        "belief_margin": round(float(belief.get("margin", 0.0)), 4),
        "scan_rows": scan_rows,
        "verify_rows": verify_rows,
        "choose_rows": choose_rows,
        "best_scan": best_scan,
        "best_verify": best_verify,
        "best_choose": best_choose,
        "best_template_choose": best_template_choose,
        "best_family": best_family,
        "best_canonical_ask_score": best_canonical_ask_score,
        "best_canonical_ask_family": best_canonical_ask_family,
        "best_scan_gain_vs_template_choose": best_scan_gain_vs_template_choose,
        "best_verify_gain_vs_template_choose": best_verify_gain_vs_template_choose,
        "best_canonical_ask_gain_vs_template_choose": best_canonical_ask_gain_vs_template_choose,
        "template_choose_optimal": template_choose_optimal,
        "positive_scan_count": sum(row["score"] > 0 for row in scan_rows),
        "positive_verify_count": sum(row["score"] > 0 for row in verify_rows),
    }


def _one_ask_successor_proxy_states(
    env: ToyTextMDP,
    initial_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    proxy_states: list[dict[str, Any]] = []
    for row in initial_snapshot["scan_rows"] + initial_snapshot["verify_rows"]:
        action = str(row["action"])
        candidate_env = env.clone()
        _observation, _reward, done, _info = candidate_env.step(action)
        if done:
            continue
        next_snapshot = _state_snapshot(candidate_env)
        remaining_ask_count = len(next_snapshot["scan_rows"]) + len(next_snapshot["verify_rows"])
        remaining_choose_count = len(next_snapshot["choose_rows"])
        if remaining_ask_count < 1 or remaining_choose_count < 1:
            continue
        best_choose_eval = (
            float(next_snapshot["best_template_choose"]["score"])
            if next_snapshot["best_template_choose"] is not None
            else None
        )
        best_scan_eval = (
            float(next_snapshot["best_scan"]["score"])
            if next_snapshot["best_scan"] is not None
            else None
        )
        best_verify_eval = (
            float(next_snapshot["best_verify"]["score"])
            if next_snapshot["best_verify"] is not None
            else None
        )
        oracle_best_ask_eval = (
            float(next_snapshot["best_canonical_ask_score"])
            if next_snapshot["best_canonical_ask_score"] is not None
            else None
        )
        oracle_ask_gain = (
            oracle_best_ask_eval - best_choose_eval
            if oracle_best_ask_eval is not None and best_choose_eval is not None
            else None
        )
        proxy_states.append(
            {
                "source_action": action,
                "source_family": _tool_family(action),
                "remaining_ask_count": remaining_ask_count,
                "remaining_choose_count": remaining_choose_count,
                "best_choose_eval": best_choose_eval,
                "best_scan_eval": best_scan_eval,
                "best_verify_eval": best_verify_eval,
                "oracle_best_ask_eval": oracle_best_ask_eval,
                "oracle_best_ask_family": str(next_snapshot["best_canonical_ask_family"]),
                "oracle_ask_gain": oracle_ask_gain,
                "direct_choose_optimal": (
                    oracle_ask_gain is None or float(oracle_ask_gain) <= 0.0
                ),
            }
        )
    return proxy_states


def _supports_family_conflict_profile(tool_profile: str) -> bool:
    return tool_profile in {
        "verify_necessary",
        "verify_necessary_family_conflict_v2",
        "verify_necessary_aggressive_stress_v3",
        "verify_necessary_aggressive_stress_v4",
    }


def _is_v2_family_conflict_profile(tool_profile: str) -> bool:
    return tool_profile == "verify_necessary_family_conflict_v2"


def _is_v3_aggressive_profile(tool_profile: str) -> bool:
    return tool_profile == "verify_necessary_aggressive_stress_v3"


def _is_v4_aggressive_profile(tool_profile: str) -> bool:
    return tool_profile == "verify_necessary_aggressive_stress_v4"


def _family_conflict_delta_for_profile(tool_profile: str) -> float:
    if _is_v4_aggressive_profile(tool_profile):
        return V4_FAMILY_CONFLICT_DELTA
    if _is_v3_aggressive_profile(tool_profile):
        return V3_FAMILY_CONFLICT_DELTA
    return FAMILY_CONFLICT_DELTA


def validate_toolized_task(task: dict[str, object]) -> dict[str, Any]:
    env = ToyTextMDP(task)
    initial = _state_snapshot(env)
    proxy_states = _one_ask_successor_proxy_states(env, initial)
    proxy_ask_gains = [
        float(snapshot["oracle_ask_gain"])
        for snapshot in proxy_states
        if snapshot.get("oracle_ask_gain") is not None
    ]
    proxy_positive_ask_gain_count = sum(
        int(float(snapshot["oracle_ask_gain"]) > 0.0)
        for snapshot in proxy_states
        if snapshot.get("oracle_ask_gain") is not None
    )
    proxy_direct_choose_optimal_count = sum(
        int(bool(snapshot.get("direct_choose_optimal")))
        for snapshot in proxy_states
    )
    tool_profile = str(task.get("tool_profile", ""))
    tool_subtype = str(task.get("tool_subtype", ""))
    family_conflict_delta = _family_conflict_delta_for_profile(tool_profile)
    passes = True
    failures: list[str] = []
    notes: list[str] = []

    if not _supports_family_conflict_profile(tool_profile):
        return {
            "task_id": str(task.get("task_id", "")),
            "tool_profile": tool_profile or "baseline",
            "tool_subtype": tool_subtype or None,
            "passes": True,
            "failures": [],
            "notes": [],
            "initial": {
                "best_family": initial["best_family"],
                "best_scan_score": initial["best_scan"]["score"] if initial["best_scan"] else None,
                "best_verify_score": initial["best_verify"]["score"] if initial["best_verify"] else None,
                "best_choose_score": initial["best_choose"]["score"] if initial["best_choose"] else None,
            },
            "after_scan": {},
            "step_local": {
                "proxy_state_count": len(proxy_states),
                "bs_positive_ask_gain_count": proxy_positive_ask_gain_count,
                "bs_positive_ask_gain_rate": _safe_div(proxy_positive_ask_gain_count, len(proxy_states)),
                "bs_direct_choose_optimal_count": proxy_direct_choose_optimal_count,
                "bs_direct_choose_optimal_rate": _safe_div(proxy_direct_choose_optimal_count, len(proxy_states)),
                "ask_gain_margin_p25": _p25(proxy_ask_gains),
                "proxy_states": proxy_states,
            },
        }

    after_scan: dict[str, Any] = {
        "snapshots": {},
        "qualifying_scan_actions": [],
        "positive_verify_after_qualifying_scan": {},
    }
    for row in initial["scan_rows"]:
        candidate_env = env.clone()
        candidate_env.step(str(row["action"]))
        next_snapshot = _state_snapshot(candidate_env)
        after_scan["snapshots"][row["action"]] = {
            "best_family": next_snapshot["best_family"],
            "best_scan_score": next_snapshot["best_scan"]["score"] if next_snapshot["best_scan"] else None,
            "best_verify_score": next_snapshot["best_verify"]["score"] if next_snapshot["best_verify"] else None,
            "best_choose_score": next_snapshot["best_choose"]["score"] if next_snapshot["best_choose"] else None,
            "best_template_choose_score": (
                next_snapshot["best_template_choose"]["score"]
                if next_snapshot["best_template_choose"]
                else None
            ),
            "best_verify_gain_vs_template_choose": next_snapshot[
                "best_verify_gain_vs_template_choose"
            ],
            "best_canonical_ask_gain_vs_template_choose": next_snapshot[
                "best_canonical_ask_gain_vs_template_choose"
            ],
            "template_choose_optimal": bool(next_snapshot["template_choose_optimal"]),
            "positive_verify_count": next_snapshot["positive_verify_count"],
            "belief_margin": next_snapshot["belief_margin"],
        }
        if (
            next_snapshot["best_verify"] is not None
            and (
                next_snapshot["best_verify_gain_vs_template_choose"] is not None
                and float(next_snapshot["best_verify_gain_vs_template_choose"]) > 0.0
            )
            and int(next_snapshot["positive_verify_count"]) >= 1
        ):
            after_scan["qualifying_scan_actions"].append(str(row["action"]))
            after_scan["positive_verify_after_qualifying_scan"][str(row["action"])] = int(
                next_snapshot["positive_verify_count"]
            )

    for proxy_state in proxy_states:
        proxy_state["task_id"] = str(task.get("task_id", ""))
        proxy_state["tool_profile"] = tool_profile
        proxy_state["tool_subtype"] = tool_subtype

    if tool_subtype == "verify_first":
        if initial["best_family"] != "verify":
            passes = False
            failures.append("initial_best_family_not_verify")
        if not initial["best_verify"] or float(initial["best_verify"]["score"]) <= 0:
            passes = False
            failures.append("initial_best_verify_not_positive")
        if (
            initial["best_verify_gain_vs_template_choose"] is None
            or float(initial["best_verify_gain_vs_template_choose"]) <= 0.0
        ):
            passes = False
            failures.append("initial_verify_not_better_than_template_choose")
        if int(initial["positive_verify_count"]) < 2:
            passes = False
            failures.append("positive_verify_count_below_two")
        notes.extend(VERIFY_NECESSARY_RELAXATION_NOTES[:1])
    elif tool_subtype == "scan_then_verify":
        if initial["best_family"] != "scan":
            passes = False
            failures.append("initial_best_family_not_scan")
        if not initial["best_scan"] or float(initial["best_scan"]["score"]) <= 0:
            passes = False
            failures.append("initial_best_scan_not_positive")
        if (
            initial["best_scan_gain_vs_template_choose"] is None
            or float(initial["best_scan_gain_vs_template_choose"]) <= 0.0
        ):
            passes = False
            failures.append("initial_scan_not_better_than_template_choose")
        if len(after_scan["qualifying_scan_actions"]) < 1:
            passes = False
            failures.append("qualifying_scan_actions_below_one")
        notes.extend(VERIFY_NECESSARY_RELAXATION_NOTES[1:2])
    elif tool_subtype == "deceptive_scan":
        if initial["best_family"] != "scan":
            passes = False
            failures.append("initial_best_family_not_scan")
        if not initial["best_scan"] or float(initial["best_scan"]["score"]) <= 0:
            passes = False
            failures.append("initial_best_scan_not_positive")
        if (
            initial["best_scan_gain_vs_template_choose"] is None
            or float(initial["best_scan_gain_vs_template_choose"]) <= 0.0
        ):
            passes = False
            failures.append("initial_scan_not_better_than_template_choose")
        deceptive_scan_proxy_states = [
            proxy_state
            for proxy_state in proxy_states
            if proxy_state.get("source_family") == "scan"
        ]
        rescue_states = [
            proxy_state
            for proxy_state in deceptive_scan_proxy_states
            if proxy_state.get("best_verify_eval") is not None
            and proxy_state.get("best_choose_eval") is not None
            and proxy_state.get("best_scan_eval") is not None
            and float(proxy_state["best_verify_eval"]) - float(proxy_state["best_choose_eval"]) >= family_conflict_delta
            and float(proxy_state["best_verify_eval"]) >= float(proxy_state["best_scan_eval"]) + family_conflict_delta
        ]
        if not rescue_states:
            passes = False
            failures.append("missing_verify_rescue_after_scan")
    else:
        passes = False
        failures.append("unknown_tool_subtype")

    notes.append(VERIFY_NECESSARY_RELAXATION_NOTES[2])
    return {
        "task_id": str(task.get("task_id", "")),
        "tool_profile": tool_profile,
        "tool_subtype": tool_subtype,
        "passes": passes,
        "failures": failures,
        "notes": notes,
        "initial": {
            "best_family": initial["best_family"],
            "best_scan_score": initial["best_scan"]["score"] if initial["best_scan"] else None,
            "best_verify_score": initial["best_verify"]["score"] if initial["best_verify"] else None,
            "best_choose_score": initial["best_choose"]["score"] if initial["best_choose"] else None,
            "best_template_choose_score": (
                initial["best_template_choose"]["score"] if initial["best_template_choose"] else None
            ),
            "best_canonical_ask_score": initial["best_canonical_ask_score"],
            "best_canonical_ask_family": initial["best_canonical_ask_family"],
            "best_scan_gain_vs_template_choose": initial["best_scan_gain_vs_template_choose"],
            "best_verify_gain_vs_template_choose": initial["best_verify_gain_vs_template_choose"],
            "best_canonical_ask_gain_vs_template_choose": initial[
                "best_canonical_ask_gain_vs_template_choose"
            ],
            "template_choose_optimal": bool(initial["template_choose_optimal"]),
            "positive_scan_count": int(initial["positive_scan_count"]),
            "positive_verify_count": int(initial["positive_verify_count"]),
            "belief_margin": initial["belief_margin"],
        },
        "after_scan": after_scan,
        "step_local": {
            "proxy_state_count": len(proxy_states),
            "bs_positive_ask_gain_count": proxy_positive_ask_gain_count,
            "bs_positive_ask_gain_rate": _safe_div(proxy_positive_ask_gain_count, len(proxy_states)),
            "bs_direct_choose_optimal_count": proxy_direct_choose_optimal_count,
            "bs_direct_choose_optimal_rate": _safe_div(proxy_direct_choose_optimal_count, len(proxy_states)),
            "ask_gain_margin_p25": _p25(proxy_ask_gains),
            "family_conflict_delta_used": round(float(family_conflict_delta), 4),
            "proxy_states": proxy_states,
        },
    }


def summarize_toolized_profile(tasks: list[dict[str, object]]) -> dict[str, Any]:
    tool_profile = str(tasks[0].get("tool_profile", "baseline")) if tasks else "baseline"
    family_conflict_delta = _family_conflict_delta_for_profile(tool_profile)
    is_v3_profile = _is_v3_aggressive_profile(tool_profile)
    is_v4_profile = _is_v4_aggressive_profile(tool_profile)
    validations = [validate_toolized_task(task) for task in tasks]
    subtype_counts = Counter(
        str(task.get("tool_subtype", ""))
        for task in tasks
        if task.get("tool_subtype")
    )
    profile_counts = Counter(
        str(task.get("tool_profile", "baseline" if task.get("toolized") else "legacy"))
        for task in tasks
    )
    initial_family_counts = Counter(
        str(validation.get("initial", {}).get("best_family", "unknown"))
        for validation in validations
    )
    pass_count = sum(int(validation["passes"]) for validation in validations)
    failure_details = [
        {
            "task_id": validation["task_id"],
            "tool_subtype": validation["tool_subtype"],
            "failures": validation["failures"],
        }
        for validation in validations
        if not validation["passes"]
    ]
    qualifying_scan_task_count = sum(
        int(bool(validation["after_scan"].get("qualifying_scan_actions")))
        for validation in validations
        if validation.get("tool_subtype") == "scan_then_verify"
    )
    initial_positive_canonical_ask_gain_count = sum(
        int(
            (
                validation.get("initial", {}).get("best_canonical_ask_gain_vs_template_choose")
                is not None
                and float(
                    validation["initial"]["best_canonical_ask_gain_vs_template_choose"]
                )
                > 0.0
            )
        )
        for validation in validations
    )
    initial_positive_scan_gain_count = sum(
        int(
            (
                validation.get("initial", {}).get("best_scan_gain_vs_template_choose") is not None
                and float(validation["initial"]["best_scan_gain_vs_template_choose"]) > 0.0
            )
        )
        for validation in validations
    )
    initial_positive_verify_gain_count = sum(
        int(
            (
                validation.get("initial", {}).get("best_verify_gain_vs_template_choose") is not None
                and float(validation["initial"]["best_verify_gain_vs_template_choose"]) > 0.0
            )
        )
        for validation in validations
    )
    initial_direct_choose_optimal_count = sum(
        int(bool(validation.get("initial", {}).get("template_choose_optimal")))
        for validation in validations
    )
    post_scan_positive_verify_gain_task_count = sum(
        int(
            any(
                snapshot.get("best_verify_gain_vs_template_choose") is not None
                and float(snapshot["best_verify_gain_vs_template_choose"]) > 0.0
                for snapshot in validation.get("after_scan", {}).get("snapshots", {}).values()
            )
        )
        for validation in validations
        if validation.get("tool_subtype") == "scan_then_verify"
    )
    direct_choose_optimal_rate = _safe_div(initial_direct_choose_optimal_count, len(tasks))
    step_local_proxy_states = [
        dict(proxy_state)
        for validation in validations
        for proxy_state in validation.get("step_local", {}).get("proxy_states", [])
    ]
    step_local_ask_gains = [
        float(proxy_state["oracle_ask_gain"])
        for proxy_state in step_local_proxy_states
        if proxy_state.get("oracle_ask_gain") is not None
    ]
    bs_positive_ask_gain_count = sum(
        int(float(proxy_state["oracle_ask_gain"]) > 0.0)
        for proxy_state in step_local_proxy_states
        if proxy_state.get("oracle_ask_gain") is not None
    )
    bs_direct_choose_optimal_count = sum(
        int(bool(proxy_state.get("direct_choose_optimal")))
        for proxy_state in step_local_proxy_states
    )
    bs_positive_ask_gain_rate = _safe_div(bs_positive_ask_gain_count, len(step_local_proxy_states))
    bs_direct_choose_optimal_rate = _safe_div(
        bs_direct_choose_optimal_count,
        len(step_local_proxy_states),
    )
    ask_gain_margin_p25 = _p25(step_local_ask_gains)
    family_conflict_proxy_states = [
        proxy_state
        for proxy_state in step_local_proxy_states
        if proxy_state.get("best_scan_eval") is not None and proxy_state.get("best_verify_eval") is not None
    ]
    family_conflict_positive_count = sum(
        int(abs(float(proxy_state["best_scan_eval"]) - float(proxy_state["best_verify_eval"])) >= family_conflict_delta)
        for proxy_state in family_conflict_proxy_states
    )
    family_conflict_positive_rate = _safe_div(
        family_conflict_positive_count,
        len(family_conflict_proxy_states),
    )
    wrong_tool_opportunity_proxy_count = sum(
        int(
            proxy_state.get("best_choose_eval") is not None
            and max(float(proxy_state["best_scan_eval"]), float(proxy_state["best_verify_eval"]))
            > float(proxy_state["best_choose_eval"])
            and abs(float(proxy_state["best_scan_eval"]) - float(proxy_state["best_verify_eval"])) >= family_conflict_delta
        )
        for proxy_state in family_conflict_proxy_states
    )
    wrong_tool_opportunity_proxy_rate = _safe_div(
        wrong_tool_opportunity_proxy_count,
        len(family_conflict_proxy_states),
    )
    scan_family_proxy_states = [
        proxy_state
        for proxy_state in step_local_proxy_states
        if proxy_state.get("tool_subtype") in {"scan_then_verify", "deceptive_scan"}
        and proxy_state.get("source_family") == "scan"
        and proxy_state.get("best_choose_eval") is not None
        and proxy_state.get("best_scan_eval") is not None
        and proxy_state.get("best_verify_eval") is not None
    ]
    scan_only_unsafe_count = sum(
        int(float(proxy_state["best_verify_eval"]) - float(proxy_state["best_choose_eval"]) >= family_conflict_delta)
        for proxy_state in scan_family_proxy_states
    )
    scan_only_unsafe_rate = _safe_div(scan_only_unsafe_count, len(scan_family_proxy_states))
    verify_rescue_after_scan_count = sum(
        int(
            float(proxy_state["best_verify_eval"]) - float(proxy_state["best_choose_eval"]) >= family_conflict_delta
            and float(proxy_state["best_verify_eval"]) >= float(proxy_state["best_scan_eval"]) + family_conflict_delta
        )
        for proxy_state in scan_family_proxy_states
    )
    verify_rescue_rate_after_scan = _safe_div(
        verify_rescue_after_scan_count,
        len(scan_family_proxy_states),
    )
    step_local_preflight_passes = (
        len(step_local_proxy_states) > 0
        and bs_positive_ask_gain_rate >= STEP_LOCAL_PREFLIGHT_POSITIVE_ASK_GAIN_MIN_RATE
        and bs_direct_choose_optimal_rate <= STEP_LOCAL_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE
        and ask_gain_margin_p25 is not None
        and float(ask_gain_margin_p25) > 0.0
    )
    task_pressure_preflight_passes = (
        len(tasks) > 0
        and initial_positive_canonical_ask_gain_count > 0
        and direct_choose_optimal_rate < TASK_PRESSURE_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE
        and (initial_positive_scan_gain_count > 0 or initial_positive_verify_gain_count > 0)
    )
    deceptive_scan_count = int(subtype_counts.get("deceptive_scan", 0))
    deceptive_scan_share = _safe_div(deceptive_scan_count, len(tasks))
    family_conflict_positive_required = family_conflict_positive_rate > 0.0
    v3_preflight_passes = (
        not is_v3_profile
        or (
            round(float(family_conflict_delta), 4) == round(float(V3_FAMILY_CONFLICT_DELTA), 4)
            and family_conflict_positive_required
            and deceptive_scan_share >= V3_DECEPTIVE_SCAN_SHARE_MIN
            and scan_only_unsafe_rate >= V3_SCAN_ONLY_UNSAFE_RATE_MIN
        )
    )
    v4_preflight_passes = (
        not is_v4_profile
        or (
            round(float(family_conflict_delta), 4) == round(float(V4_FAMILY_CONFLICT_DELTA), 4)
            and family_conflict_positive_rate > V4_FAMILY_CONFLICT_POSITIVE_RATE_MIN
            and deceptive_scan_share >= V4_DECEPTIVE_SCAN_SHARE_MIN
            and scan_only_unsafe_rate >= V4_SCAN_ONLY_UNSAFE_RATE_MIN
            and wrong_tool_opportunity_proxy_rate > 0.0
            and verify_rescue_rate_after_scan > 0.0
        )
    )
    combined_preflight_passes = (
        task_pressure_preflight_passes
        and step_local_preflight_passes
        and v3_preflight_passes
        and v4_preflight_passes
    )
    return {
        "task_count": len(tasks),
        "pass_count": pass_count,
        "failure_count": len(tasks) - pass_count,
        "tool_profile_counts": dict(profile_counts),
        "tool_subtype_counts": dict(subtype_counts),
        "initial_family_counts": dict(initial_family_counts),
        "verify_better": int(initial_family_counts.get("verify", 0)),
        "scan_better": int(initial_family_counts.get("scan", 0)),
        "qualifying_scan_task_count": qualifying_scan_task_count,
        "initial_positive_canonical_ask_gain_count": initial_positive_canonical_ask_gain_count,
        "initial_positive_scan_gain_count": initial_positive_scan_gain_count,
        "initial_positive_verify_gain_count": initial_positive_verify_gain_count,
        "initial_direct_choose_optimal_count": initial_direct_choose_optimal_count,
        "initial_direct_choose_optimal_rate": round(float(direct_choose_optimal_rate), 4),
        "post_scan_positive_verify_gain_task_count": post_scan_positive_verify_gain_task_count,
        "task_pressure_preflight_passes": task_pressure_preflight_passes,
        "step_local_proxy_state_count": len(step_local_proxy_states),
        "bs_positive_ask_gain_count": bs_positive_ask_gain_count,
        "bs_positive_ask_gain_rate": round(float(bs_positive_ask_gain_rate), 4),
        "bs_direct_choose_optimal_count": bs_direct_choose_optimal_count,
        "bs_direct_choose_optimal_rate": round(float(bs_direct_choose_optimal_rate), 4),
        "ask_gain_margin_p25": None if ask_gain_margin_p25 is None else round(float(ask_gain_margin_p25), 4),
        "family_conflict_delta": round(float(family_conflict_delta), 4),
        "family_conflict_delta_used": round(float(family_conflict_delta), 4),
        "family_conflict_proxy_state_count": len(family_conflict_proxy_states),
        "family_conflict_positive_count": family_conflict_positive_count,
        "family_conflict_positive_rate": round(float(family_conflict_positive_rate), 4),
        "wrong_tool_opportunity_proxy_count": wrong_tool_opportunity_proxy_count,
        "wrong_tool_opportunity_proxy_rate": round(float(wrong_tool_opportunity_proxy_rate), 4),
        "scan_family_proxy_state_count": len(scan_family_proxy_states),
        "scan_only_unsafe_count": scan_only_unsafe_count,
        "scan_only_unsafe_rate": round(float(scan_only_unsafe_rate), 4),
        "verify_rescue_after_scan_count": verify_rescue_after_scan_count,
        "verify_rescue_rate_after_scan": round(float(verify_rescue_rate_after_scan), 4),
        "deceptive_scan_count": deceptive_scan_count,
        "deceptive_scan_share": round(float(deceptive_scan_share), 4),
        "step_local_preflight_passes": step_local_preflight_passes,
        "v3_preflight_passes": bool(v3_preflight_passes),
        "v4_preflight_passes": bool(v4_preflight_passes),
        "combined_preflight_passes": bool(combined_preflight_passes),
        "task_pressure_preflight_notes": [
            "positive canonical ask gain must appear at least once",
            "template direct choose cannot remain optimal for nearly all generated tasks",
            "at least one ask family must beat template direct choose in some initial states",
        ],
        "step_local_preflight_notes": [
            "one-ask successor proxy states are offline-generated approximations, not live blocked-step replays",
            "direct choose is evaluated against the template-choose proxy inside those one-ask successor states",
            f"bs_positive_ask_gain_rate must be >= {STEP_LOCAL_PREFLIGHT_POSITIVE_ASK_GAIN_MIN_RATE:.2f}",
            f"bs_direct_choose_optimal_rate must be <= {STEP_LOCAL_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE:.2f}",
            "ask_gain_margin_p25 must be strictly positive",
        ],
        "stress_metric_notes": [
            "family_conflict_positive_rate is measured on proxy states where both scan and verify remain available",
            f"delta={family_conflict_delta:.2f} is the minimum substantive score gap for family conflict, unsafe scan-only, and verify rescue",
            "proxy scores use the cumulative reward scale of canonical [ASK_x, CHOOSE_BEST] rollouts under evaluate_candidate()",
            "verify_rescue_rate_after_scan is reported only on scan_then_verify and deceptive_scan proxy states reached after an initial scan action",
        ],
        "report_only_stress_metrics": [
            "family_conflict_positive_rate",
            "wrong_tool_opportunity_proxy_rate",
            "verify_rescue_rate_after_scan",
        ],
        "v3_preflight_notes": (
            [
                "family_conflict_delta_used must equal 0.20",
                "family_conflict_positive_rate must be > 0",
                "deceptive_scan_share must be >= 0.50",
                "scan_only_unsafe_rate must be >= 0.30",
            ]
            if is_v3_profile
            else []
        ),
        "v4_preflight_notes": (
            [
                "family_conflict_delta_used must equal 0.20",
                "family_conflict_positive_rate must be > 0.75",
                "deceptive_scan_share must be >= 0.50",
                "scan_only_unsafe_rate must be >= 0.50",
                "wrong_tool_opportunity_proxy_rate must be > 0",
                "verify_rescue_rate_after_scan must be > 0",
            ]
            if is_v4_profile
            else []
        ),
        "failure_details": failure_details[:10],
        "constraint_relaxations": list(VERIFY_NECESSARY_RELAXATION_NOTES),
    }


def _verify_necessary_subtype_schedule(count: int, rng: random.Random) -> list[str]:
    target_verify_first = round(count * 0.4)
    target_scan_then_verify = count - target_verify_first
    subtypes = ["verify_first"] * target_verify_first + ["scan_then_verify"] * target_scan_then_verify
    rng.shuffle(subtypes)
    return subtypes


def _family_conflict_v2_subtype_schedule(count: int, rng: random.Random) -> list[str]:
    target_verify_first = round(count * 0.4)
    target_scan_then_verify = round(count * 0.4)
    target_deceptive_scan = count - target_verify_first - target_scan_then_verify
    subtypes = (
        ["verify_first"] * target_verify_first
        + ["scan_then_verify"] * target_scan_then_verify
        + ["deceptive_scan"] * target_deceptive_scan
    )
    rng.shuffle(subtypes)
    return subtypes


def _aggressive_stress_v3_subtype_schedule(count: int, rng: random.Random) -> list[str]:
    target_deceptive_scan = round(count * 0.50)
    target_verify_first = round((count - target_deceptive_scan) * 0.50)
    target_scan_then_verify = count - target_deceptive_scan - target_verify_first
    subtypes = (
        ["verify_first"] * target_verify_first
        + ["scan_then_verify"] * target_scan_then_verify
        + ["deceptive_scan"] * target_deceptive_scan
    )
    rng.shuffle(subtypes)
    return subtypes


def _aggressive_stress_v4_subtype_schedule(count: int, rng: random.Random) -> list[str]:
    target_deceptive_scan = round(count * 0.60)
    target_verify_first = round((count - target_deceptive_scan) * 0.50)
    target_scan_then_verify = count - target_deceptive_scan - target_verify_first
    subtypes = (
        ["verify_first"] * target_verify_first
        + ["scan_then_verify"] * target_scan_then_verify
        + ["deceptive_scan"] * target_deceptive_scan
    )
    rng.shuffle(subtypes)
    return subtypes


def _build_verify_necessary_task(
    *,
    task_id: str,
    difficulty: str,
    noise_level: float,
    ask_cost: float,
    answer: str,
    seed_value: int,
    rng: random.Random,
    extras: dict[str, object] | None,
    tool_profile: str,
    tool_subtype: str,
) -> dict[str, object]:
    clue_count = (
        6
        if tool_profile
        in {
            "verify_necessary_family_conflict_v2",
            "verify_necessary_aggressive_stress_v3",
            "verify_necessary_aggressive_stress_v4",
        }
        else 5
    )
    for _attempt in range(200):
        if tool_profile == "verify_necessary_family_conflict_v2":
            if tool_subtype == "verify_first":
                clues = _verify_first_tool_clues_v2(
                    answer=answer,
                    ask_cost=ask_cost,
                    clue_count=clue_count,
                    rng=rng,
                )
            elif tool_subtype == "scan_then_verify":
                clues = _scan_then_verify_tool_clues_v2(
                    answer=answer,
                    ask_cost=ask_cost,
                    clue_count=clue_count,
                    rng=rng,
                )
            else:
                clues = _deceptive_scan_tool_clues(
                    answer=answer,
                    ask_cost=ask_cost,
                    clue_count=clue_count,
                    rng=rng,
                )
        elif tool_profile in {
            "verify_necessary_aggressive_stress_v3",
            "verify_necessary_aggressive_stress_v4",
        }:
            if tool_subtype == "verify_first":
                clues = _verify_first_tool_clues_v3(
                    answer=answer,
                    ask_cost=ask_cost,
                    clue_count=clue_count,
                    rng=rng,
                )
            elif tool_subtype == "scan_then_verify":
                clues = _scan_then_verify_tool_clues_v3(
                    answer=answer,
                    ask_cost=ask_cost,
                    clue_count=clue_count,
                    rng=rng,
                )
            else:
                clues = _deceptive_scan_tool_clues_v3(
                    answer=answer,
                    ask_cost=ask_cost,
                    clue_count=clue_count,
                    rng=rng,
                )
        elif tool_subtype == "verify_first":
            clues = _verify_first_tool_clues(
                answer=answer,
                ask_cost=ask_cost,
                clue_count=clue_count,
                rng=rng,
            )
        else:
            clues = _scan_then_verify_tool_clues(
                answer=answer,
                ask_cost=ask_cost,
                clue_count=clue_count,
                rng=rng,
            )
        task: dict[str, object] = {
            "task_id": task_id,
            "answer": answer,
            "difficulty": difficulty,
            "noise_level": noise_level,
            "ask_cost": ask_cost,
            "max_steps": clue_count + 1,
            "seed": seed_value,
            "clues": clues,
            "toolized": True,
            "tool_profile": tool_profile,
            "tool_subtype": tool_subtype,
        }
        if extras:
            task.update(extras)
        validation = validate_toolized_task(task)
        if validation["passes"]:
            return task
    raise RuntimeError(f"Failed to generate a valid verify_necessary task for {task_id}")


def _build_task(
    *,
    task_id: str,
    difficulty: str,
    noise_level: float,
    ask_cost: float,
    answer: str,
    seed_value: int,
    rng: random.Random,
    extras: dict[str, object] | None = None,
    toolized: bool = False,
    toolized_profile: str = "baseline",
    tool_subtype: str | None = None,
) -> dict[str, object]:
    clue_count = 3 if difficulty != "hard" else 4
    if toolized and _supports_family_conflict_profile(toolized_profile):
        return _build_verify_necessary_task(
            task_id=task_id,
            difficulty=difficulty,
            noise_level=noise_level,
            ask_cost=ask_cost,
            answer=answer,
            seed_value=seed_value,
            rng=rng,
            extras=extras,
            tool_profile=toolized_profile,
            tool_subtype=str(tool_subtype or "verify_first"),
        )

    if toolized:
        clues = _baseline_tool_clues(
            answer=answer,
            noise_level=noise_level,
            ask_cost=ask_cost,
            clue_count=clue_count,
            rng=rng,
        )
    else:
        clues = []
        for clue_id in range(1, clue_count + 1):
            scores = _build_scores(answer, noise_level, rng)
            reliability = round(max(0.45, 1.0 - noise_level + rng.uniform(-0.05, 0.05)), 2)
            clues.append(
                {
                    "id": clue_id,
                    "scores": scores,
                    "reliability": reliability,
                    "text": _format_clue(clue_id, scores, reliability),
                }
            )

    task: dict[str, object] = {
        "task_id": task_id,
        "answer": answer,
        "difficulty": difficulty,
        "noise_level": noise_level,
        "ask_cost": ask_cost,
        "max_steps": clue_count + 1,
        "seed": seed_value,
        "clues": clues,
        "toolized": toolized,
    }
    if toolized:
        task["tool_profile"] = toolized_profile
    if extras:
        task.update(extras)
    return task


def generate_tasks(
    count: int = 200,
    seed: int = 7,
    *,
    toolized: bool = False,
    toolized_profile: str = "baseline",
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    tasks: list[dict[str, object]] = []

    difficulties = {"easy": 0.05, "medium": 0.18, "hard": 0.32}
    ask_costs = {"easy": -0.03, "medium": -0.05, "hard": -0.08}
    subtype_schedule = (
        _verify_necessary_subtype_schedule(count, rng)
        if toolized and toolized_profile == "verify_necessary"
        else _family_conflict_v2_subtype_schedule(count, rng)
        if toolized and toolized_profile == "verify_necessary_family_conflict_v2"
        else _aggressive_stress_v3_subtype_schedule(count, rng)
        if toolized and toolized_profile == "verify_necessary_aggressive_stress_v3"
        else _aggressive_stress_v4_subtype_schedule(count, rng)
        if toolized and toolized_profile == "verify_necessary_aggressive_stress_v4"
        else [None] * count
    )

    for index in range(count):
        difficulty = rng.choices(
            population=list(difficulties.keys()),
            weights=(0.35, 0.45, 0.20),
            k=1,
        )[0]
        tasks.append(
            _build_task(
                task_id=f"task-{index + 1:04d}",
                difficulty=difficulty,
                noise_level=difficulties[difficulty],
                ask_cost=ask_costs[difficulty],
                answer=(
                    VERIFY_NECESSARY_PRESSURE_ANSWER
                    if toolized and _supports_family_conflict_profile(toolized_profile)
                    else rng.choice(CHOICES)
                ),
                seed_value=seed * 1000 + index,
                rng=rng,
                extras={"benchmark_family": "single_episode"},
                toolized=toolized,
                toolized_profile=toolized_profile,
                tool_subtype=subtype_schedule[index],
            )
        )
    return tasks


def _task_numeric_id(task: dict[str, object]) -> int:
    task_id = str(task.get("task_id", ""))
    suffix = task_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _repair_target_subtype_counts(selected_task_count: int) -> Counter[str]:
    target_verify_first = round(selected_task_count * 0.4)
    target_scan_then_verify = selected_task_count - target_verify_first
    return Counter(
        {
            "verify_first": target_verify_first,
            "scan_then_verify": target_scan_then_verify,
        }
    )


def _repair_objective(
    *,
    positive_count: int,
    direct_choose_optimal_count: int,
    proxy_state_count: int,
) -> tuple[int, int, int, int, int]:
    required_positive = math.ceil(
        proxy_state_count * STEP_LOCAL_PREFLIGHT_POSITIVE_ASK_GAIN_MIN_RATE
    )
    allowed_direct_choose = math.floor(
        proxy_state_count * STEP_LOCAL_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE
    )
    positive_deficit = max(0, required_positive - positive_count)
    direct_excess = max(0, direct_choose_optimal_count - allowed_direct_choose)
    return (
        positive_deficit + direct_excess,
        positive_deficit,
        direct_excess,
        direct_choose_optimal_count,
        -positive_count,
    )


def _reference_repair_rank_key(item: dict[str, Any]) -> tuple[int, int, float, int]:
    return (
        int(item["direct_count"]),
        -int(item["positive_count"]),
        -float(item["ask_gain_margin_p25"]),
        int(item["pool_index"]),
    )


def _select_reference_repair_tasks(
    *,
    candidate_tasks: list[dict[str, object]],
    selected_task_count: int,
    selection_policy: str,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    if selection_policy not in REFERENCE_REPAIR_SELECTION_POLICIES:
        raise ValueError(f"Unsupported repair selection policy: {selection_policy}")
    if selected_task_count <= 0:
        raise ValueError("selected_task_count must be positive")
    if len(candidate_tasks) < selected_task_count:
        raise ValueError("candidate pool must be at least as large as selected_task_count")

    validations = [validate_toolized_task(task) for task in candidate_tasks]
    items: list[dict[str, Any]] = []
    for pool_index, (task, validation) in enumerate(zip(candidate_tasks, validations, strict=True)):
        step_local = validation.get("step_local", {})
        items.append(
            {
                "task": task,
                "validation": validation,
                "pool_index": pool_index,
                "task_id_num": _task_numeric_id(task),
                "tool_subtype": str(task.get("tool_subtype", "")),
                "positive_count": int(step_local.get("bs_positive_ask_gain_count", 0)),
                "direct_count": int(step_local.get("bs_direct_choose_optimal_count", 0)),
                "proxy_count": int(step_local.get("proxy_state_count", 0)),
                "ask_gain_margin_p25": float(step_local.get("ask_gain_margin_p25") or 0.0),
            }
        )

    target_counts = _repair_target_subtype_counts(selected_task_count)
    selected_by_subtype: dict[str, list[dict[str, Any]]] = {}
    reserve_by_subtype: dict[str, list[dict[str, Any]]] = {}
    for subtype, target_count in target_counts.items():
        subtype_items = [
            item
            for item in items
            if item["tool_subtype"] == subtype and item["validation"]["passes"]
        ]
        subtype_items.sort(key=_reference_repair_rank_key)
        if len(subtype_items) < target_count:
            raise RuntimeError(
                f"Insufficient {subtype} tasks to satisfy reference repair quota: "
                f"need {target_count}, have {len(subtype_items)}"
            )
        selected_by_subtype[subtype] = subtype_items[:target_count]
        reserve_by_subtype[subtype] = subtype_items[target_count:]

    selected_items = [
        item
        for subtype in sorted(selected_by_subtype)
        for item in selected_by_subtype[subtype]
    ]
    proxy_state_count = sum(int(item["proxy_count"]) for item in selected_items)
    positive_count = sum(int(item["positive_count"]) for item in selected_items)
    direct_choose_optimal_count = sum(int(item["direct_count"]) for item in selected_items)
    current_objective = _repair_objective(
        positive_count=positive_count,
        direct_choose_optimal_count=direct_choose_optimal_count,
        proxy_state_count=proxy_state_count,
    )

    improved = True
    while current_objective[0] > 0 and improved:
        improved = False
        best_swap: dict[str, Any] | None = None
        best_key: tuple[Any, ...] | None = None
        for subtype in sorted(target_counts):
            selected_candidates = selected_by_subtype[subtype]
            reserve_candidates = reserve_by_subtype[subtype]
            for selected_index, selected_item in enumerate(selected_candidates):
                for reserve_index, reserve_item in enumerate(reserve_candidates):
                    new_positive = (
                        positive_count
                        - int(selected_item["positive_count"])
                        + int(reserve_item["positive_count"])
                    )
                    new_direct = (
                        direct_choose_optimal_count
                        - int(selected_item["direct_count"])
                        + int(reserve_item["direct_count"])
                    )
                    new_objective = _repair_objective(
                        positive_count=new_positive,
                        direct_choose_optimal_count=new_direct,
                        proxy_state_count=proxy_state_count,
                    )
                    candidate_key = (
                        new_objective,
                        -float(reserve_item["ask_gain_margin_p25"]),
                        int(reserve_item["pool_index"]),
                        int(selected_item["pool_index"]),
                    )
                    if best_key is None or candidate_key < best_key:
                        best_key = candidate_key
                        best_swap = {
                            "subtype": subtype,
                            "selected_index": selected_index,
                            "reserve_index": reserve_index,
                            "new_positive": new_positive,
                            "new_direct": new_direct,
                            "new_objective": new_objective,
                        }
        if best_swap and best_swap["new_objective"] < current_objective:
            subtype = str(best_swap["subtype"])
            selected_index = int(best_swap["selected_index"])
            reserve_index = int(best_swap["reserve_index"])
            selected_item = selected_by_subtype[subtype][selected_index]
            reserve_item = reserve_by_subtype[subtype][reserve_index]
            selected_by_subtype[subtype][selected_index] = reserve_item
            reserve_by_subtype[subtype][reserve_index] = selected_item
            positive_count = int(best_swap["new_positive"])
            direct_choose_optimal_count = int(best_swap["new_direct"])
            current_objective = best_swap["new_objective"]
            improved = True

    final_selected_items = [
        item
        for subtype in sorted(selected_by_subtype)
        for item in selected_by_subtype[subtype]
    ]
    final_selected_items.sort(key=lambda item: (int(item["task_id_num"]), str(item["task"]["task_id"])))
    selected_tasks = [item["task"] for item in final_selected_items]
    metadata = {
        "selection_policy": selection_policy,
        "candidate_pool_count": len(candidate_tasks),
        "selected_task_count": len(selected_tasks),
        "discarded_task_count": len(candidate_tasks) - len(selected_tasks),
    }
    return selected_tasks, metadata


def _arc_profile(style: str) -> tuple[dict[str, float], dict[str, str]]:
    if style == "cautious":
        return (
            {
                "ask_if_uncertain": 0.82,
                "justify_costly_questions": 0.58,
                "honor_prior_commitments": 0.72,
            },
            {
                "priority": "consistency",
                "description": "Recognition favors caution that is still consistent across the arc.",
            },
        )
    if style == "decisive":
        return (
            {
                "ask_if_uncertain": 0.52,
                "justify_costly_questions": 0.86,
                "honor_prior_commitments": 0.66,
            },
            {
                "priority": "decisiveness",
                "description": "Recognition favors closing once the reason is stable enough to own.",
            },
        )
    return (
        {
            "ask_if_uncertain": 0.68,
            "justify_costly_questions": 0.72,
            "honor_prior_commitments": 0.74,
        },
        {
            "priority": "justification",
            "description": "Recognition favors a balanced reason that can justify both asking and choosing.",
        },
    )


def generate_linked_tasks(
    *,
    arc_count: int = 40,
    episodes_per_arc: int = 3,
    seed: int = 7,
    toolized: bool = False,
    toolized_profile: str = "baseline",
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    tasks: list[dict[str, object]] = []
    difficulties = {"easy": 0.06, "medium": 0.18, "hard": 0.30}
    ask_costs = {"easy": -0.03, "medium": -0.05, "hard": -0.08}
    total_tasks = arc_count * episodes_per_arc
    subtype_schedule = (
        _verify_necessary_subtype_schedule(total_tasks, rng)
        if toolized and toolized_profile == "verify_necessary"
        else _family_conflict_v2_subtype_schedule(total_tasks, rng)
        if toolized and toolized_profile == "verify_necessary_family_conflict_v2"
        else _aggressive_stress_v3_subtype_schedule(total_tasks, rng)
        if toolized and toolized_profile == "verify_necessary_aggressive_stress_v3"
        else _aggressive_stress_v4_subtype_schedule(total_tasks, rng)
        if toolized and toolized_profile == "verify_necessary_aggressive_stress_v4"
        else [None] * total_tasks
    )
    subtype_index = 0

    for arc_index in range(arc_count):
        style = rng.choice(ARC_STYLES)
        norm_bank, recognition_check = _arc_profile(style)
        arc_id = f"arc-{arc_index + 1:03d}"
        role = rng.choice(("field_analyst", "reviewer", "planner"))

        for episode_index in range(1, episodes_per_arc + 1):
            difficulty = rng.choices(
                population=list(difficulties.keys()),
                weights=(0.25, 0.45, 0.30) if episode_index > 1 else (0.35, 0.45, 0.20),
                k=1,
            )[0]
            revisit_required = episode_index > 1
            extras: dict[str, object] = {
                "benchmark_family": "linked_episode",
                "arc_id": arc_id,
                "episode_index": episode_index,
                "identity_context": {
                    "style": style,
                    "role": role,
                    "long_horizon": "maintain a defensible decision style across repeated tasks",
                },
                "norm_bank": norm_bank,
                "recognition_check": recognition_check,
                "commitment_revisit": {
                    "required": revisit_required,
                    "description": (
                        "Revisit the previous arc commitment and explain whether it should be preserved or revised."
                        if revisit_required
                        else "Establish an initial commitment that later arc episodes can revisit."
                    ),
                },
            }
            tasks.append(
                _build_task(
                    task_id=f"{arc_id}-ep-{episode_index:02d}",
                    difficulty=difficulty,
                    noise_level=difficulties[difficulty],
                    ask_cost=ask_costs[difficulty],
                    answer=(
                        VERIFY_NECESSARY_PRESSURE_ANSWER
                        if toolized and _supports_family_conflict_profile(toolized_profile)
                        else rng.choice(CHOICES)
                    ),
                    seed_value=seed * 100000 + arc_index * 10 + episode_index,
                    rng=rng,
                    extras=extras,
                    toolized=toolized,
                    toolized_profile=toolized_profile,
                    tool_subtype=subtype_schedule[subtype_index],
                )
            )
            subtype_index += 1
    return tasks


def generate_tasks_file(
    path: Path,
    count: int = 200,
    seed: int = 7,
    *,
    linked_arcs: int = 0,
    episodes_per_arc: int = 3,
    toolized: bool = False,
    toolized_profile: str = "baseline",
    allow_preflight_failure: bool = False,
    preflight_summary_out: Path | None = None,
    reference_repair: bool = False,
    repair_pool_count: int | None = None,
    repair_selection_policy: str = "minimal_preflight_margin_fix",
    source_failed_attempt: Path | None = None,
    repair_attempt_index: int = 1,
) -> dict[str, Any] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] | None = None
    repair_metadata: dict[str, Any] = {}
    if reference_repair:
        if allow_preflight_failure:
            raise ValueError(
                "reference repair does not allow --allow-preflight-failure; "
                "failed attempts must stop before hash generation and live execution."
            )
        if linked_arcs > 0:
            raise ValueError("reference repair is not supported for linked-arc generation")
        if not toolized or toolized_profile != "verify_necessary":
            raise ValueError(
                "reference repair requires --toolized --toolized-profile verify_necessary"
            )
        if repair_pool_count is None:
            raise ValueError("reference repair requires repair_pool_count")
        if repair_pool_count < count:
            raise ValueError("repair_pool_count must be >= count")
        candidate_tasks = generate_tasks(
            count=repair_pool_count,
            seed=seed,
            toolized=toolized,
            toolized_profile=toolized_profile,
        )
        tasks, selection_metadata = _select_reference_repair_tasks(
            candidate_tasks=candidate_tasks,
            selected_task_count=count,
            selection_policy=repair_selection_policy,
        )
        repair_metadata = {
            "repair_attempt": True,
            "repair_lane": REFERENCE_REPAIR_LANE,
            "source_failed_attempt": (
                str(source_failed_attempt) if source_failed_attempt is not None else None
            ),
            "selection_policy": repair_selection_policy,
            "candidate_pool_count": int(selection_metadata["candidate_pool_count"]),
            "selected_task_count": int(selection_metadata["selected_task_count"]),
            "discarded_task_count": int(selection_metadata["discarded_task_count"]),
            "repair_attempt_index": int(repair_attempt_index),
        }
    elif linked_arcs > 0:
        tasks = generate_linked_tasks(
            arc_count=linked_arcs,
            episodes_per_arc=episodes_per_arc,
            seed=seed,
            toolized=toolized,
            toolized_profile=toolized_profile,
        )
    else:
        tasks = generate_tasks(
            count=count,
            seed=seed,
            toolized=toolized,
            toolized_profile=toolized_profile,
        )
    summary: dict[str, Any] | None = None
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    if toolized and _supports_family_conflict_profile(toolized_profile):
        summary = summarize_toolized_profile(tasks)
        if repair_metadata:
            summary.update(repair_metadata)
        if preflight_summary_out is not None:
            preflight_summary_out.parent.mkdir(parents=True, exist_ok=True)
            preflight_summary_out.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if (
            not allow_preflight_failure
            and not summary["combined_preflight_passes"]
        ):
            raise RuntimeError(
                f"{toolized_profile} preflight failed; regenerate the task file instead of "
                "proceeding to live evaluation."
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ToyTextMDP tasks.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--linked-arcs",
        type=int,
        default=0,
        help="Generate linked-episode benchmark arcs instead of standalone tasks.",
    )
    parser.add_argument("--episodes-per-arc", type=int, default=3)
    parser.add_argument(
        "--toolized",
        action="store_true",
        help="Emit ASK_SCAN/ASK_VERIFY clue variants instead of legacy ASK_CLUE tasks.",
    )
    parser.add_argument(
        "--toolized-profile",
        choices=TOOLIZED_PROFILES,
        default="baseline",
        help="Profile used when --toolized is enabled.",
    )
    parser.add_argument(
        "--allow-preflight-failure",
        action="store_true",
        help="Write the task file and summary even when local preflight fails.",
    )
    parser.add_argument(
        "--preflight-summary-out",
        type=Path,
        help="Optional path to write the local preflight summary JSON.",
    )
    parser.add_argument(
        "--reference-repair",
        action="store_true",
        help="Regenerate a verify_necessary reference pack from a deterministic candidate pool without relaxing hard gates.",
    )
    parser.add_argument(
        "--repair-pool-count",
        type=int,
        help="Candidate pool size for reference repair generation.",
    )
    parser.add_argument(
        "--repair-selection-policy",
        choices=REFERENCE_REPAIR_SELECTION_POLICIES,
        default="minimal_preflight_margin_fix",
        help="Selection policy used when --reference-repair is enabled.",
    )
    parser.add_argument(
        "--source-failed-attempt",
        type=Path,
        help="Path to the failed reference attempt directory being repaired.",
    )
    parser.add_argument(
        "--repair-attempt-index",
        type=int,
        default=1,
        help="1-based repair attempt index for reference repair summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_tasks_file(
        args.out,
        count=args.count,
        seed=args.seed,
        linked_arcs=args.linked_arcs,
        episodes_per_arc=args.episodes_per_arc,
        toolized=args.toolized,
        toolized_profile=args.toolized_profile,
        allow_preflight_failure=args.allow_preflight_failure,
        preflight_summary_out=args.preflight_summary_out,
        reference_repair=args.reference_repair,
        repair_pool_count=args.repair_pool_count,
        repair_selection_policy=args.repair_selection_policy,
        source_failed_attempt=args.source_failed_attempt,
        repair_attempt_index=args.repair_attempt_index,
    )
    if summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
