from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_battery_summary import _load_pack


LOWER_IS_BETTER_METRICS = (
    "choose_swallow_rate_given_oracle_tool_choice",
    "selected_vs_oracle_family_mismatch_rate",
)
HIGHER_IS_BETTER_METRICS = ("verify_usage_rate",)
MODE_ORDER = ("full", "selector_penalty", "voi_memory_v1")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_trigger_good(stress_pack: dict[str, Any], delta: float) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    full = (stress_pack.get("modes") or {}).get("full") or {}
    voi = (stress_pack.get("modes") or {}).get("voi_memory_v1") or {}
    if (voi.get("tool_choice_step_count") or 0) <= 0 or (voi.get("oracle_tool_choice_step_count") or 0) <= 0:
        return findings

    for metric in LOWER_IS_BETTER_METRICS:
        full_value = _as_float(full.get(metric))
        voi_value = _as_float(voi.get(metric))
        if full_value is None or voi_value is None:
            continue
        improvement = full_value - voi_value
        if improvement >= delta:
            findings.append(
                {
                    "metric": metric,
                    "direction": "voi_memory_v1_lower_than_full",
                    "full_value": full_value,
                    "voi_memory_v1_value": voi_value,
                    "delta_value": improvement,
                }
            )

    full_verify = _as_float(full.get("verify_usage_rate"))
    voi_verify = _as_float(voi.get("verify_usage_rate"))
    if full_verify is not None and voi_verify is not None:
        improvement = voi_verify - full_verify
        if improvement >= delta:
            findings.append(
                {
                    "metric": "verify_usage_rate",
                    "direction": "voi_memory_v1_higher_than_full",
                    "full_value": full_verify,
                    "voi_memory_v1_value": voi_verify,
                    "delta_value": improvement,
                }
            )
    return findings


def _build_negative_inversion(stress_pack: dict[str, Any], delta: float) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    selector = (stress_pack.get("modes") or {}).get("selector_penalty") or {}
    voi = (stress_pack.get("modes") or {}).get("voi_memory_v1") or {}

    for metric in LOWER_IS_BETTER_METRICS:
        selector_value = _as_float(selector.get(metric))
        voi_value = _as_float(voi.get(metric))
        if selector_value is None or voi_value is None:
            continue
        deterioration = voi_value - selector_value
        if deterioration >= delta:
            findings.append(
                {
                    "metric": metric,
                    "direction": "voi_memory_v1_higher_than_selector_penalty",
                    "selector_penalty_value": selector_value,
                    "voi_memory_v1_value": voi_value,
                    "delta_value": deterioration,
                }
            )

    selector_verify = _as_float(selector.get("verify_usage_rate"))
    voi_verify = _as_float(voi.get("verify_usage_rate"))
    if selector_verify is not None and voi_verify is not None:
        deterioration = selector_verify - voi_verify
        if deterioration >= delta:
            findings.append(
                {
                    "metric": "verify_usage_rate",
                    "direction": "voi_memory_v1_lower_than_selector_penalty",
                    "selector_penalty_value": selector_verify,
                    "voi_memory_v1_value": voi_verify,
                    "delta_value": deterioration,
                }
            )
    return findings


def build_stress_probe_summary(*, run_root: Path, delta: float = 0.10) -> dict[str, Any]:
    stress_pack = _load_pack(run_root)
    blocked = stress_pack.get("blocked") or {}
    trigger_good = _build_trigger_good(stress_pack, delta)
    negative_inversion = _build_negative_inversion(stress_pack, delta)
    pure_task_pressure = (
        blocked.get("route_target") == "task_pressure"
        and int(blocked.get("task_pressure", 0)) == int(blocked.get("blocked_step_count", 0))
        and int(blocked.get("blocked_step_count", 0)) > 0
    )

    separation_reopened = bool(trigger_good)

    if negative_inversion:
        route = "C"
        next_action = "pause_for_gate_memory_selector_review"
        verdict = "negative_inversion"
    elif separation_reopened:
        route = "B"
        next_action = "prepare_seed11_reference_confirmation_review"
        verdict = "separation_reopened"
    elif pure_task_pressure:
        route = "A"
        next_action = "enter_stronger_v3_patch"
        verdict = "same_design_collapse"
    else:
        route = "manual"
        next_action = "manual_review_required"
        verdict = "unresolved"

    return {
        "delta": delta,
        "stress_pack": stress_pack,
        "trigger_good": trigger_good,
        "trigger_good_count": len(trigger_good),
        "negative_inversion": negative_inversion,
        "negative_inversion_count": len(negative_inversion),
        "pure_task_pressure": pure_task_pressure,
        "separation_reopened": separation_reopened,
        "route": route,
        "verdict": verdict,
        "next_action": next_action,
    }


def render_stress_probe_summary(summary: dict[str, Any]) -> str:
    stress = summary["stress_pack"]
    lines = [
        "# Seed11 Stress Probe Summary",
        "",
        f"- delta: {summary['delta']:.2f}",
        f"- verdict: {summary['verdict']}",
        f"- route: {summary['route']}",
        f"- next_action: {summary['next_action']}",
        f"- receipt_ok: {'true' if stress.get('receipt_ok') else 'false'}",
        f"- pure_task_pressure: {'true' if summary['pure_task_pressure'] else 'false'}",
        f"- separation_reopened: {'true' if summary['separation_reopened'] else 'false'}",
        f"- negative_inversion_count: {summary['negative_inversion_count']}",
        "",
        "## Trigger Good",
        "",
    ]
    if summary["trigger_good"]:
        for item in summary["trigger_good"]:
            lines.append(
                f"- {item['metric']} / {item['direction']} / full={item['full_value']:.4f} / "
                f"voi_memory_v1={item['voi_memory_v1_value']:.4f} / delta={item['delta_value']:.4f}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Negative Inversion", ""])
    if summary["negative_inversion"]:
        for item in summary["negative_inversion"]:
            lines.append(
                f"- {item['metric']} / {item['direction']} / selector_penalty={item['selector_penalty_value']:.4f} / "
                f"voi_memory_v1={item['voi_memory_v1_value']:.4f} / delta={item['delta_value']:.4f}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Stress Pack Metrics",
            "",
            f"- run_name: {stress.get('run_name')}",
            f"- run_root: {stress.get('run_root')}",
            f"- blocked_step: task_pressure={(stress.get('blocked') or {}).get('task_pressure', 0)}, "
            f"surfaced_ask_deficit={(stress.get('blocked') or {}).get('surfaced_ask_deficit', 0)}, "
            f"hard_cutoff_candidate={(stress.get('blocked') or {}).get('hard_cutoff_candidate', 0)}, "
            f"route_target={(stress.get('blocked') or {}).get('route_target', 'unknown')}",
            "",
            "| mode | tool_choice_step_count | oracle_tool_choice_step_count | realized_tool_choice_coverage | choose_swallow | selected_vs_oracle_family_mismatch | tool_family_mismatch | verify_usage_rate | wrong_tool_rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode_name in MODE_ORDER:
        mode = (stress.get("modes") or {}).get(mode_name) or {}
        lines.append(
            f"| {mode_name} | {mode.get('tool_choice_step_count', 0):.1f} | "
            f"{mode.get('oracle_tool_choice_step_count', 0):.1f} | "
            f"{(mode.get('realized_tool_choice_coverage') or 0):.4f} | "
            f"{(mode.get('choose_swallow_rate_given_oracle_tool_choice') or 0):.4f} | "
            f"{(mode.get('selected_vs_oracle_family_mismatch_rate') or 0):.4f} | "
            f"{(mode.get('tool_family_mismatch_rate_vs_oracle') or 0):.4f} | "
            f"{(mode.get('verify_usage_rate') or 0):.4f} | "
            f"{(mode.get('wrong_tool_rate') or 0):.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a seed-level stress-pack-only summary.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_stress_probe_summary(run_root=args.run_root, delta=args.delta)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_stress_probe_summary(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
