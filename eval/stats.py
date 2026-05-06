from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

EXPERIMENTAL_MODES = (
    "selector_penalty",
    "voi_coverage_only",
    "voi_gate_only",
    "voi_memory_v1",
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def paired_t_test(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    n = min(len(xs), len(ys))
    if n < 2:
        return {"mean_diff": None, "t_stat": None}
    diffs = [x - y for x, y in zip(xs[:n], ys[:n], strict=True)]
    avg = mean(diffs)
    variance = sum((diff - avg) ** 2 for diff in diffs) / (n - 1)
    if variance == 0:
        return {"mean_diff": avg, "t_stat": None}
    return {"mean_diff": avg, "t_stat": avg / math.sqrt(variance / n)}


def permutation_test(
    xs: list[float],
    ys: list[float],
    *,
    iterations: int = 5000,
    seed: int = 7,
) -> dict[str, float]:
    n = min(len(xs), len(ys))
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0}

    pairs = list(zip(xs[:n], ys[:n], strict=True))
    observed = mean(x - y for x, y in pairs)
    rng = random.Random(seed)
    extreme = 0

    for _ in range(iterations):
        statistic = mean(
            (x - y) if rng.random() < 0.5 else (y - x)
            for x, y in pairs
        )
        if abs(statistic) >= abs(observed):
            extreme += 1

    return {"observed_diff": observed, "p_value": (extreme + 1) / (iterations + 1)}


def bootstrap_interval(
    values: list[float],
    *,
    iterations: int = 3000,
    seed: int = 7,
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        samples.append(mean(sample))
    samples.sort()
    low = samples[int(0.025 * len(samples))]
    high = samples[int(0.975 * len(samples))]
    return {"mean": mean(values), "ci_low": low, "ci_high": high}


def one_way_repeated_measures_anova(groups: dict[str, list[float]]) -> dict[str, float | None]:
    labels = list(groups)
    if len(labels) < 3:
        return {"f_stat": None}

    min_len = min(len(groups[label]) for label in labels)
    if min_len < 2:
        return {"f_stat": None}

    truncated = {label: values[:min_len] for label, values in groups.items()}
    grand_mean = mean(value for values in truncated.values() for value in values)
    condition_means = {label: mean(values) for label, values in truncated.items()}
    subject_means = [mean(truncated[label][index] for label in labels) for index in range(min_len)]

    ss_conditions = min_len * sum((condition_means[label] - grand_mean) ** 2 for label in labels)
    ss_subjects = len(labels) * sum((subject_mean - grand_mean) ** 2 for subject_mean in subject_means)
    ss_total = sum((value - grand_mean) ** 2 for values in truncated.values() for value in values)
    ss_error = ss_total - ss_conditions - ss_subjects

    df_conditions = len(labels) - 1
    df_error = (len(labels) - 1) * (min_len - 1)
    if df_conditions <= 0 or df_error <= 0 or ss_error <= 0:
        return {"f_stat": None}
    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error
    return {"f_stat": ms_conditions / ms_error if ms_error else None}


def _enabled_tests(raw: str) -> set[str]:
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _comparison_bundle(
    xs: list[float],
    ys: list[float],
    *,
    enabled_tests: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "ttest" in enabled_tests:
        result["ttest"] = paired_t_test(xs, ys)
    if "permutation" in enabled_tests:
        result["permutation"] = permutation_test(xs, ys)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight statistics on metrics.json.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--tests", type=str, default="permutation,ttest,anova")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    series = metrics["series"]
    enabled = _enabled_tests(args.tests)

    results: dict[str, Any] = {
        "reward_full_vs_ctrl": _comparison_bundle(
            series["full_rewards"],
            series["ctrl_rewards"],
            enabled_tests=enabled,
        ),
        "reward_full_vs_perm": _comparison_bundle(
            series["full_rewards"],
            series["perm_rewards"],
            enabled_tests=enabled,
        ),
        "success_full_vs_ctrl": _comparison_bundle(
            series["full_success"],
            series["ctrl_success"],
            enabled_tests=enabled,
        ),
        "success_full_vs_perm": _comparison_bundle(
            series["full_success"],
            series["perm_success"],
            enabled_tests=enabled,
        ),
        "steps_full_vs_ctrl": _comparison_bundle(
            series["full_steps"],
            series["ctrl_steps"],
            enabled_tests=enabled,
        ),
        "steps_full_vs_perm": _comparison_bundle(
            series["full_steps"],
            series["perm_steps"],
            enabled_tests=enabled,
        ),
        "DCR": bootstrap_interval(list(series["dcr_by_task"].values())),
        "CUR": bootstrap_interval(list(series["cur_by_task"].values())),
        "SRI": bootstrap_interval(series["sri_contributions"]),
    }

    for mode_name in EXPERIMENTAL_MODES:
        reward_key = f"{mode_name}_rewards"
        success_key = f"{mode_name}_success"
        steps_key = f"{mode_name}_steps"
        if reward_key not in series:
            continue
        results[f"reward_{mode_name}_vs_full"] = _comparison_bundle(
            series[reward_key],
            series["full_rewards"],
            enabled_tests=enabled,
        )
        results[f"success_{mode_name}_vs_full"] = _comparison_bundle(
            series[success_key],
            series["full_success"],
            enabled_tests=enabled,
        )
        results[f"steps_{mode_name}_vs_full"] = _comparison_bundle(
            series[steps_key],
            series["full_steps"],
            enabled_tests=enabled,
        )

    if "anova" in enabled:
        results["anova_rewards"] = one_way_repeated_measures_anova(
            {
                "full": series["full_rewards"],
                "ctrl": series["ctrl_rewards"],
                "perm": series["perm_rewards"],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
