from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_battery_summary import _load_pack


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seed11_trigger_good(stress_pack: dict[str, Any], delta: float) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    full = (stress_pack.get("modes") or {}).get("full") or {}
    selector = (stress_pack.get("modes") or {}).get("selector_penalty") or {}
    voi = (stress_pack.get("modes") or {}).get("voi_memory_v1") or {}
    if (voi.get("tool_choice_step_count") or 0) <= 0 or (voi.get("oracle_tool_choice_step_count") or 0) <= 0:
        return findings

    lower_better = (
        "choose_swallow_rate_given_oracle_tool_choice",
        "selected_vs_oracle_family_mismatch_rate",
    )
    for metric in lower_better:
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

    negative_inversion: list[dict[str, Any]] = []
    for metric in lower_better:
        selector_value = _as_float(selector.get(metric))
        voi_value = _as_float(voi.get(metric))
        if selector_value is None or voi_value is None:
            continue
        deterioration = voi_value - selector_value
        if deterioration >= delta:
            negative_inversion.append(
                {
                    "metric": metric,
                    "direction": "voi_memory_v1_higher_than_selector_penalty",
                    "selector_penalty_value": selector_value,
                    "voi_memory_v1_value": voi_value,
                    "delta_value": deterioration,
                }
            )
    selector_verify = _as_float(selector.get("verify_usage_rate"))
    if selector_verify is not None and voi_verify is not None:
        deterioration = selector_verify - voi_verify
        if deterioration >= delta:
            negative_inversion.append(
                {
                    "metric": "verify_usage_rate",
                    "direction": "voi_memory_v1_lower_than_selector_penalty",
                    "selector_penalty_value": selector_verify,
                    "voi_memory_v1_value": voi_verify,
                    "delta_value": deterioration,
                }
            )
    return findings, negative_inversion


def build_combined_summary_v2(
    *,
    seed7_battery_summary_path: Path,
    seed11_stress_run_root: Path,
    delta: float = 0.10,
) -> dict[str, Any]:
    seed7_summary = json.loads(seed7_battery_summary_path.read_text(encoding="utf-8"))
    seed11_stress = _load_pack(seed11_stress_run_root)
    seed11_trigger_good, seed11_negative_inversion = _seed11_trigger_good(seed11_stress, delta)

    blocked = seed11_stress.get("blocked") or {}
    seed11_pure_task_pressure = (
        blocked.get("route_target") == "task_pressure"
        and int(blocked.get("task_pressure", 0)) == int(blocked.get("blocked_step_count", 0))
        and int(blocked.get("blocked_step_count", 0)) > 0
    )
    seed7_no_positive = not bool(seed7_summary.get("trigger_good"))
    seed7_no_inversion = not bool(seed7_summary.get("negative_inversion"))
    seed11_no_positive = not bool(seed11_trigger_good)
    seed11_no_inversion = not bool(seed11_negative_inversion)

    stable_negative_result = bool(
        seed7_summary.get("stress", {}).get("receipt_ok")
        and seed11_stress.get("receipt_ok")
        and seed7_no_positive
        and seed7_no_inversion
        and seed11_no_positive
        and seed11_no_inversion
        and seed11_pure_task_pressure
    )
    negative_inversion_detected = bool(seed11_negative_inversion)
    stress_only_conditional_effect = bool(
        seed11_stress.get("receipt_ok")
        and len(seed11_trigger_good) >= 2
        and not negative_inversion_detected
    )

    if negative_inversion_detected:
        next_action = "pause_for_ask_gate_memory_diagnosis"
        verdict = "negative_inversion_detected"
    elif stable_negative_result:
        next_action = "enter_v3_aggressive_stress"
        verdict = "stable_negative_result"
    elif stress_only_conditional_effect:
        next_action = "prepare_seed11_reference_confirmation_review"
        verdict = "stress_only_conditional_effect"
    else:
        next_action = "manual_review_required"
        verdict = "unresolved"

    return {
        "delta": delta,
        "seed7": seed7_summary,
        "seed11_stress": seed11_stress,
        "seed11_trigger_good": seed11_trigger_good,
        "seed11_negative_inversion": seed11_negative_inversion,
        "stable_negative_result": stable_negative_result,
        "stress_only_conditional_effect": stress_only_conditional_effect,
        "negative_inversion_detected": negative_inversion_detected,
        "next_action": next_action,
        "verdict": verdict,
    }


def render_combined_summary_v2(summary: dict[str, Any]) -> str:
    lines = [
        "# Combined Battery Summary V2",
        "",
        f"- delta: {summary['delta']:.2f}",
        f"- verdict: {summary['verdict']}",
        f"- next_action: {summary['next_action']}",
        f"- stable_negative_result: {'true' if summary['stable_negative_result'] else 'false'}",
        f"- stress_only_conditional_effect: {'true' if summary['stress_only_conditional_effect'] else 'false'}",
        f"- negative_inversion_detected: {'true' if summary['negative_inversion_detected'] else 'false'}",
        "",
        "## Seed7 Battery",
        "",
        f"- next_action: {summary['seed7'].get('next_action')}",
        f"- seed11_trigger: {'true' if summary['seed7'].get('seed11_trigger') else 'false'}",
        f"- trigger_good_count: {len(summary['seed7'].get('trigger_good', []))}",
        f"- negative_inversion_count: {len(summary['seed7'].get('negative_inversion', []))}",
        "",
        "## Seed11 Stress Probe",
        "",
        f"- run_root: {summary['seed11_stress'].get('run_root')}",
        f"- receipt_ok: {'true' if summary['seed11_stress'].get('receipt_ok') else 'false'}",
        f"- route_target: {(summary['seed11_stress'].get('blocked') or {}).get('route_target', 'unknown')}",
        f"- blocked_step_count: {(summary['seed11_stress'].get('blocked') or {}).get('blocked_step_count', 0)}",
        "",
        "## Seed11 Trigger Good",
        "",
    ]
    if summary["seed11_trigger_good"]:
        for item in summary["seed11_trigger_good"]:
            lines.append(
                f"- {item['metric']} / {item['direction']} / full={item['full_value']:.4f} / "
                f"voi_memory_v1={item['voi_memory_v1_value']:.4f} / delta={item['delta_value']:.4f}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Seed11 Negative Inversion", ""])
    if summary["seed11_negative_inversion"]:
        for item in summary["seed11_negative_inversion"]:
            lines.append(
                f"- {item['metric']} / {item['direction']} / selector_penalty={item['selector_penalty_value']:.4f} / "
                f"voi_memory_v1={item['voi_memory_v1_value']:.4f} / delta={item['delta_value']:.4f}"
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined seed7+seed11 summary for V2 routing.")
    parser.add_argument("--seed7-battery-summary", type=Path, required=True)
    parser.add_argument("--seed11-stress-run-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_combined_summary_v2(
        seed7_battery_summary_path=args.seed7_battery_summary,
        seed11_stress_run_root=args.seed11_stress_run_root,
        delta=args.delta,
    )
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_combined_summary_v2(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
