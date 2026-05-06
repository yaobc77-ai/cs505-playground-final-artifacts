from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.diagnose_reflection import run_diagnosis
from eval.stats import bootstrap_interval, mean

SUMMARY_FIELDS = ("success_rate", "avg_reward", "avg_steps")
PROCESS_FIELDS = ("avg_asks_per_episode", "choose_without_asking_rate", "reflection_admission_rate")
EXPERIMENTAL_MODE_LABELS = {
    "selector_penalty": "selector_penalty",
    "voi_coverage_only": "voi_coverage_only",
    "voi_gate_only": "voi_gate_only",
    "voi_memory_v1": "voi_memory_v1",
}
PAIRWISE_KEYS = (
    "reward_full_vs_ctrl",
    "reward_full_vs_perm",
    "success_full_vs_ctrl",
    "success_full_vs_perm",
    "steps_full_vs_ctrl",
    "steps_full_vs_perm",
    *tuple(
        key
        for mode_name in EXPERIMENTAL_MODE_LABELS
        for key in (
            f"reward_{mode_name}_vs_full",
            f"success_{mode_name}_vs_full",
            f"steps_{mode_name}_vs_full",
        )
    ),
)
MAIN_MODE_LABELS = {
    "full": "full",
    "ctrl": "no_reflection",
    "perm": "permute",
}
SUPPLEMENTARY_MODE_LABELS = {
    "no_self_model": "no_self_model",
    "no_cf_eval": "no_cf_eval",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_seeds(raw: str) -> list[int]:
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def _aggregate_scalar(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    return bootstrap_interval(values)


def _aggregate_metric_block(
    metrics_by_seed: dict[int, dict[str, Any]],
    *,
    section_name: str,
    mode_names: tuple[str, ...],
    field_names: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for mode_name in mode_names:
        mode_block: dict[str, dict[str, float]] = {}
        for field_name in field_names:
            values = [
                float(metrics[section_name][mode_name][field_name])
                for metrics in metrics_by_seed.values()
                if mode_name in metrics.get(section_name, {})
            ]
            mode_block[field_name] = _aggregate_scalar(values)
        if mode_block:
            aggregate[mode_name] = mode_block
    return aggregate


def _aggregate_top_level_metrics(metrics_by_seed: dict[int, dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in ("DCR", "CUR", "SRI"):
        values = [
            float(metrics["summary"][metric_name])
            for metrics in metrics_by_seed.values()
            if metric_name in metrics.get("summary", {})
        ]
        aggregate[metric_name] = _aggregate_scalar(values)
    return aggregate


def _aggregate_subgroups(metrics_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    group_names = sorted(
        {
            group_name
            for metrics in metrics_by_seed.values()
            for group_name in metrics.get("subgroup_summary", {})
        }
    )

    for group_name in group_names:
        aggregate[group_name] = {}
        subgroup_values = sorted(
            {
                subgroup_value
                for metrics in metrics_by_seed.values()
                for subgroup_value in metrics.get("subgroup_summary", {}).get(group_name, {})
            }
        )
        for subgroup_value in subgroup_values:
            aggregate[group_name][subgroup_value] = {}
            for mode_name in ("full", "ctrl", "perm"):
                aggregate[group_name][subgroup_value][mode_name] = {}
                for field_name in SUMMARY_FIELDS:
                    values = [
                        float(metrics["subgroup_summary"][group_name][subgroup_value][mode_name][field_name])
                        for metrics in metrics_by_seed.values()
                        if subgroup_value in metrics.get("subgroup_summary", {}).get(group_name, {})
                        and mode_name in metrics["subgroup_summary"][group_name][subgroup_value]
                    ]
                    aggregate[group_name][subgroup_value][mode_name][field_name] = _aggregate_scalar(values)
    return aggregate


def _aggregate_pairwise(stats_by_seed: dict[int, dict[str, Any]], key: str) -> dict[str, Any]:
    mean_diffs: list[float] = []
    t_stats: list[float] = []
    p_values: list[float] = []

    for stats in stats_by_seed.values():
        comparison = stats.get(key, {})
        ttest = comparison.get("ttest", {})
        permutation = comparison.get("permutation", {})

        mean_diff = ttest.get("mean_diff")
        if mean_diff is None:
            mean_diff = permutation.get("observed_diff")
        if mean_diff is not None:
            mean_diffs.append(float(mean_diff))

        t_stat = ttest.get("t_stat")
        if t_stat is not None:
            t_stats.append(float(t_stat))

        p_value = permutation.get("p_value")
        if p_value is not None:
            p_values.append(float(p_value))

    if not mean_diffs and not t_stats and not p_values:
        return {}

    result: dict[str, Any] = {
        "mean_diff": _aggregate_scalar(mean_diffs),
        "seed_count": len(mean_diffs),
    }
    if t_stats:
        result["t_stat"] = _aggregate_scalar(t_stats)
    if p_values:
        result["permutation_p_value"] = {
            "mean": mean(p_values),
            "min": min(p_values),
            "max": max(p_values),
        }
        result["significant_seed_count"] = sum(1 for value in p_values if value < 0.05)
    return result


def _aggregate_stats(stats_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "comparisons": {
            key: value
            for key in PAIRWISE_KEYS
            if (value := _aggregate_pairwise(stats_by_seed, key))
        },
        "mechanism": {},
    }

    for metric_name in ("DCR", "CUR", "SRI"):
        values = [
            float(stats[metric_name]["mean"])
            for stats in stats_by_seed.values()
            if metric_name in stats
        ]
        aggregate["mechanism"][metric_name] = _aggregate_scalar(values)

    anova_values = [
        float(stats["anova_rewards"]["f_stat"])
        for stats in stats_by_seed.values()
        if stats.get("anova_rewards", {}).get("f_stat") is not None
    ]
    if anova_values:
        aggregate["anova_rewards"] = _aggregate_scalar(anova_values)

    return aggregate


def _fmt_interval(interval: dict[str, float], digits: int = 3) -> str:
    return f"{interval['mean']:.{digits}f} [{interval['ci_low']:.{digits}f}, {interval['ci_high']:.{digits}f}]"


def _fmt_p_value(block: dict[str, Any]) -> str:
    p_block = block.get("permutation_p_value")
    if not p_block:
        return "n/a"
    return f"{p_block['mean']:.4f} (sig {block.get('significant_seed_count', 0)}/{block.get('seed_count', 0)})"


def _render_report_summary(
    *,
    seeds: list[int],
    aggregate_metrics: dict[str, Any],
    aggregate_stats: dict[str, Any],
) -> str:
    lines = [
        "# Multi-seed Experiment Summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Main Results",
        "",
        "| Mode | Success rate | Avg reward | Avg steps | Avg asks | Choose w/o asking | Reflection admitted |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for mode_name, label in MAIN_MODE_LABELS.items():
        summary = aggregate_metrics["summary"][mode_name]
        process = aggregate_metrics["process_summary"][mode_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt_interval(summary["success_rate"]),
                    _fmt_interval(summary["avg_reward"]),
                    _fmt_interval(summary["avg_steps"]),
                    _fmt_interval(process["avg_asks_per_episode"]),
                    _fmt_interval(process["choose_without_asking_rate"]),
                    _fmt_interval(process["reflection_admission_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Main Statistics",
            "",
        ]
    )
    comparison_labels = {
        "reward_full_vs_ctrl": "Reward: full vs no_reflection",
        "reward_full_vs_perm": "Reward: full vs permute",
        "success_full_vs_ctrl": "Success: full vs no_reflection",
        "success_full_vs_perm": "Success: full vs permute",
        "steps_full_vs_ctrl": "Steps: full vs no_reflection",
        "steps_full_vs_perm": "Steps: full vs permute",
    }
    for mode_name, label in EXPERIMENTAL_MODE_LABELS.items():
        comparison_labels[f"reward_{mode_name}_vs_full"] = f"Reward: {label} vs full"
        comparison_labels[f"success_{mode_name}_vs_full"] = f"Success: {label} vs full"
        comparison_labels[f"steps_{mode_name}_vs_full"] = f"Steps: {label} vs full"
    for key, block in aggregate_stats["comparisons"].items():
        lines.append(
            f"- {comparison_labels[key]}: diff={_fmt_interval(block['mean_diff'])}; permutation p={_fmt_p_value(block)}"
        )

    lines.extend(
        [
            f"- DCR: {_fmt_interval(aggregate_metrics['summary']['DCR'])}",
            f"- CUR: {_fmt_interval(aggregate_metrics['summary']['CUR'])}",
            f"- SRI: {_fmt_interval(aggregate_metrics['summary']['SRI'])}",
            "",
            "## Supplementary Ablations",
            "",
            "| Mode | Success rate | Avg reward | Avg steps | Avg asks | Choose w/o asking | Reflection admitted |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for mode_name, label in SUPPLEMENTARY_MODE_LABELS.items():
        summary = aggregate_metrics["supplementary_summary"].get(mode_name)
        process = aggregate_metrics["process_summary"].get(mode_name)
        if not summary or not process:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    _fmt_interval(summary["success_rate"]),
                    _fmt_interval(summary["avg_reward"]),
                    _fmt_interval(summary["avg_steps"]),
                    _fmt_interval(process["avg_asks_per_episode"]),
                    _fmt_interval(process["choose_without_asking_rate"]),
                    _fmt_interval(process["reflection_admission_rate"]),
                ]
            )
            + " |"
        )

    experimental_summary = aggregate_metrics.get("experimental_summary", {})
    if experimental_summary:
        lines.extend(
            [
                "",
                "## Experimental Modes",
                "",
                "| Mode | Success rate | Avg reward | Avg steps | Avg asks | Choose w/o asking | Reflection admitted |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for mode_name, label in EXPERIMENTAL_MODE_LABELS.items():
            summary = experimental_summary.get(mode_name)
            process = aggregate_metrics["process_summary"].get(mode_name)
            if not summary or not process:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        _fmt_interval(summary["success_rate"]),
                        _fmt_interval(summary["avg_reward"]),
                        _fmt_interval(summary["avg_steps"]),
                        _fmt_interval(process["avg_asks_per_episode"]),
                        _fmt_interval(process["choose_without_asking_rate"]),
                        _fmt_interval(process["reflection_admission_rate"]),
                    ]
                )
                + " |"
            )

    for group_name, title in (("difficulty", "Difficulty breakdown"), ("noise_level", "Noise breakdown")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Group | Full reward | No reflection reward | Permute reward | Full success | No reflection success | Permute success |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        subgroup_block = aggregate_metrics["subgroup_summary"].get(group_name, {})
        for subgroup_value, subgroup_summary in subgroup_block.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        subgroup_value,
                        _fmt_interval(subgroup_summary["full"]["avg_reward"]),
                        _fmt_interval(subgroup_summary["ctrl"]["avg_reward"]),
                        _fmt_interval(subgroup_summary["perm"]["avg_reward"]),
                        _fmt_interval(subgroup_summary["full"]["success_rate"]),
                        _fmt_interval(subgroup_summary["ctrl"]["success_rate"]),
                        _fmt_interval(subgroup_summary["perm"]["success_rate"]),
                    ]
                )
                + " |"
            )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment outputs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--with-diagnosis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    run_root = args.run_root

    metrics_by_seed: dict[int, dict[str, Any]] = {}
    stats_by_seed: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        seed_root = run_root / f"seed_{seed}"
        metrics_path = seed_root / "metrics.json"
        stats_path = seed_root / "stats.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics.json for seed {seed}: {metrics_path}")
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing stats.json for seed {seed}: {stats_path}")
        metrics_by_seed[seed] = _load_json(metrics_path)
        stats_by_seed[seed] = _load_json(stats_path)

    aggregate_metrics = {
        "config": {
            "run_root": str(run_root),
            "seeds": seeds,
            "seed_count": len(seeds),
        },
        "summary": {
            **_aggregate_metric_block(
                metrics_by_seed,
                section_name="summary",
                mode_names=("full", "ctrl", "perm"),
                field_names=SUMMARY_FIELDS,
            ),
            **_aggregate_top_level_metrics(metrics_by_seed),
        },
        "supplementary_summary": _aggregate_metric_block(
            metrics_by_seed,
            section_name="supplementary_summary",
            mode_names=("no_self_model", "no_cf_eval"),
            field_names=SUMMARY_FIELDS,
        ),
        "experimental_summary": _aggregate_metric_block(
            metrics_by_seed,
            section_name="experimental_summary",
            mode_names=tuple(EXPERIMENTAL_MODE_LABELS),
            field_names=SUMMARY_FIELDS,
        ),
        "process_summary": _aggregate_metric_block(
            metrics_by_seed,
            section_name="process_summary",
            mode_names=("full", "ctrl", "perm", "no_self_model", "no_cf_eval", *tuple(EXPERIMENTAL_MODE_LABELS)),
            field_names=PROCESS_FIELDS,
        ),
        "subgroup_summary": _aggregate_subgroups(metrics_by_seed),
    }
    aggregate_stats = {
        "config": {
            "run_root": str(run_root),
            "seeds": seeds,
            "seed_count": len(seeds),
        },
        **_aggregate_stats(stats_by_seed),
    }
    report_summary = _render_report_summary(
        seeds=seeds,
        aggregate_metrics=aggregate_metrics,
        aggregate_stats=aggregate_stats,
    )

    (run_root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "aggregate_stats.json").write_text(
        json.dumps(aggregate_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_root / "report_summary.md").write_text(report_summary, encoding="utf-8")

    if args.with_diagnosis:
        diagnosis, diagnosis_report = run_diagnosis(run_root=run_root, seeds=seeds)
        (run_root / "diagnosis.json").write_text(
            json.dumps(diagnosis, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_root / "diagnosis_report.md").write_text(
            diagnosis_report,
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
