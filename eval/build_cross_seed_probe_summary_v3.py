from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRACKED_MODE = "voi_memory_v1"
TRACKED_METRICS = (
    "tool_choice_step_count",
    "oracle_tool_choice_step_count",
    "realized_tool_choice_coverage",
    "choose_swallow_rate_given_oracle_tool_choice",
    "selected_vs_oracle_family_mismatch_rate",
    "tool_family_mismatch_rate_vs_oracle",
    "verify_usage_rate",
    "wrong_tool_rate",
)


def _mode_metrics(payload: dict[str, Any], pack_key: str) -> dict[str, Any]:
    return (
        payload.get(pack_key, {})
        .get("modes", {})
        .get(TRACKED_MODE, {})
    )


def build_cross_seed_probe_summary_v3(
    *,
    seed7_battery_summary_path: Path,
    seed11_stress_summary_path: Path,
    delta: float = 0.10,
) -> dict[str, Any]:
    seed7 = json.loads(seed7_battery_summary_path.read_text(encoding="utf-8"))
    seed11 = json.loads(seed11_stress_summary_path.read_text(encoding="utf-8"))
    seed7_stress = _mode_metrics(seed7, "stress")
    seed11_stress = _mode_metrics(seed11, "stress_pack")

    choose_swallow_delta = None
    if (
        seed7_stress.get("choose_swallow_rate_given_oracle_tool_choice") is not None
        and seed11_stress.get("choose_swallow_rate_given_oracle_tool_choice") is not None
    ):
        choose_swallow_delta = (
            float(seed11_stress["choose_swallow_rate_given_oracle_tool_choice"])
            - float(seed7_stress["choose_swallow_rate_given_oracle_tool_choice"])
        )

    choose_swallow_collapse_shape_persists = (
        choose_swallow_delta is not None and abs(choose_swallow_delta) < delta
    )

    if seed11.get("negative_inversion_count", 0) > 0:
        route = "C"
        next_action = "pause_for_gate_memory_selector_review"
        verdict = "negative_inversion"
    elif seed11.get("separation_reopened"):
        route = "B"
        next_action = "prepare_seed11_reference_confirmation_review"
        verdict = "seed11_reopened_separation"
    elif seed11.get("pure_task_pressure"):
        route = "A"
        next_action = "enter_stronger_v3_patch"
        verdict = "stable_same_design_collapse"
    else:
        route = "manual"
        next_action = "manual_review_required"
        verdict = "unresolved"

    return {
        "delta": delta,
        "seed7_battery_summary_path": str(seed7_battery_summary_path),
        "seed11_stress_summary_path": str(seed11_stress_summary_path),
        "seed11_receipt_ok": bool(seed11.get("stress_pack", {}).get("receipt_ok")),
        "seed11_choose_swallow_collapse_shape_persists": choose_swallow_collapse_shape_persists,
        "seed11_choose_swallow_delta_vs_seed7_stress": choose_swallow_delta,
        "seed11_pure_task_pressure": bool(seed11.get("pure_task_pressure")),
        "seed11_negative_inversion_count": int(seed11.get("negative_inversion_count", 0)),
        "seed11_separation_reopened": bool(seed11.get("separation_reopened")),
        "seed11_trigger_good_count": int(seed11.get("trigger_good_count", 0)),
        "route": route,
        "verdict": verdict,
        "next_action": next_action,
        "seed7_stress_voi_memory_v1": {metric: seed7_stress.get(metric) for metric in TRACKED_METRICS},
        "seed11_stress_voi_memory_v1": {metric: seed11_stress.get(metric) for metric in TRACKED_METRICS},
    }


def render_cross_seed_probe_summary_v3(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-Seed Probe Summary V3",
        "",
        f"- delta: {summary['delta']:.2f}",
        f"- verdict: {summary['verdict']}",
        f"- route: {summary['route']}",
        f"- next_action: {summary['next_action']}",
        f"- seed11_receipt_ok: {'true' if summary['seed11_receipt_ok'] else 'false'}",
        f"- seed11_choose_swallow_collapse_shape_persists: {'true' if summary['seed11_choose_swallow_collapse_shape_persists'] else 'false'}",
        f"- seed11_pure_task_pressure: {'true' if summary['seed11_pure_task_pressure'] else 'false'}",
        f"- seed11_negative_inversion_count: {summary['seed11_negative_inversion_count']}",
        f"- seed11_separation_reopened: {'true' if summary['seed11_separation_reopened'] else 'false'}",
        f"- seed11_trigger_good_count: {summary['seed11_trigger_good_count']}",
        "",
        "## `voi_memory_v1` Seed7 vs Seed11 Stress",
        "",
        "| metric | seed7_stress | seed11_stress |",
        "| --- | ---: | ---: |",
    ]
    seed7_metrics = summary["seed7_stress_voi_memory_v1"]
    seed11_metrics = summary["seed11_stress_voi_memory_v1"]
    for metric in TRACKED_METRICS:
        seed7_value = seed7_metrics.get(metric)
        seed11_value = seed11_metrics.get(metric)
        if isinstance(seed7_value, (int, float)):
            left = f"{seed7_value:.4f}"
        else:
            left = str(seed7_value)
        if isinstance(seed11_value, (int, float)):
            right = f"{seed11_value:.4f}"
        else:
            right = str(seed11_value)
        lines.append(f"| {metric} | {left} | {right} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cross-seed v3 probe summary.")
    parser.add_argument("--seed7-battery-summary", type=Path, required=True)
    parser.add_argument("--seed11-stress-summary", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_cross_seed_probe_summary_v3(
        seed7_battery_summary_path=args.seed7_battery_summary,
        seed11_stress_summary_path=args.seed11_stress_summary,
        delta=args.delta,
    )
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_cross_seed_probe_summary_v3(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
