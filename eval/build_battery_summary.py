from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOWER_IS_BETTER_METRICS = (
    "choose_swallow_rate_given_oracle_tool_choice",
    "selected_vs_oracle_family_mismatch_rate",
)
HIGHER_IS_BETTER_METRICS = ("verify_usage_rate",)
PRIMARY_COMPARISON_MODE = "voi_memory_v1"
PRIMARY_METRICS = (
    "tool_choice_step_count",
    "oracle_tool_choice_step_count",
    "realized_tool_choice_coverage",
    "choose_swallow_rate_given_oracle_tool_choice",
    "selected_vs_oracle_family_mismatch_rate",
    "tool_family_mismatch_rate_vs_oracle",
    "verify_usage_rate",
    "wrong_tool_rate",
)
MODE_ORDER = ("full", "selector_penalty", "voi_memory_v1")


def _parse_key_value_markdown(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _receipt_ok(receipt: dict[str, str]) -> bool:
    return (
        receipt.get("run_completed", "").lower() == "true"
        and receipt.get("jsonl_integrity") == "pass"
        and receipt.get("task_id_pairing") == "pass"
        and receipt.get("safe_for_interpretation", "").lower() == "true"
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_mode_metrics(diagnosis: dict[str, Any], mode_name: str) -> dict[str, float | None]:
    mode = diagnosis.get("modes", {}).get(mode_name, {})
    overall = mode.get("overall", {})
    episodes = mode.get("episodes", {})
    return {
        "tool_choice_step_count": _as_float(overall.get("tool_choice_step_count")),
        "oracle_tool_choice_step_count": _as_float(overall.get("oracle_tool_choice_step_count")),
        "realized_tool_choice_coverage": _as_float(
            (
                (overall.get("tool_choice_step_count") or 0)
                / (overall.get("oracle_tool_choice_step_count") or 1)
            )
            if overall.get("oracle_tool_choice_step_count")
            else None
        ),
        "choose_swallow_rate_given_oracle_tool_choice": _as_float(
            overall.get("choose_swallow_rate_given_oracle_tool_choice")
        ),
        "selected_vs_oracle_family_mismatch_rate": _as_float(
            overall.get("selected_vs_oracle_family_mismatch_rate")
        ),
        "tool_family_mismatch_rate_vs_oracle": _as_float(
            overall.get("tool_family_mismatch_rate_vs_oracle")
        ),
        "verify_usage_rate": _as_float(overall.get("verify_usage_rate")),
        "wrong_tool_rate": _as_float(overall.get("wrong_tool_rate")),
        "success_rate": _as_float(episodes.get("success_rate")),
        "blocked_step_count": _as_float(
            (diagnosis.get("blocked_step_audit", {}).get("voi_memory_v1") or {}).get("blocked_step_count")
        ),
    }


def _load_pack(run_root: Path) -> dict[str, Any]:
    receipt_path = run_root / "POST_RUN_RECEIPT.md"
    diagnosis_path = run_root / "diagnosis.json"
    blocked_path = run_root / "blocked_step_audit.md"
    for path in (receipt_path, diagnosis_path, blocked_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing battery artifact: {path}")

    receipt = _parse_key_value_markdown(receipt_path)
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    blocked = (diagnosis.get("blocked_step_audit", {}).get("voi_memory_v1") or {})

    return {
        "run_root": str(run_root),
        "run_name": receipt.get("run_name", run_root.name),
        "receipt_ok": _receipt_ok(receipt),
        "receipt": {
            "run_completed": receipt.get("run_completed", "false"),
            "jsonl_integrity": receipt.get("jsonl_integrity", "fail"),
            "task_id_pairing": receipt.get("task_id_pairing", "fail"),
            "safe_for_interpretation": receipt.get("safe_for_interpretation", "false"),
        },
        "modes": {
            mode_name: _load_mode_metrics(diagnosis, mode_name)
            for mode_name in MODE_ORDER
            if mode_name in diagnosis.get("modes", {})
        },
        "blocked": {
            "task_pressure": blocked.get("task_pressure_count", 0),
            "surfaced_ask_deficit": blocked.get("surfaced_ask_deficit_count", 0),
            "hard_cutoff_candidate": blocked.get("hard_cutoff_candidate_count", 0),
            "blocked_step_count": blocked.get("blocked_step_count", 0),
            "route_target": blocked.get("route_target", "unknown"),
        },
    }


def _build_trigger_good(
    *,
    reference_modes: dict[str, dict[str, float | None]],
    stress_modes: dict[str, dict[str, float | None]],
    delta: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_findings: list[dict[str, Any]] = []
    sanity_findings: list[dict[str, Any]] = []
    for mode_name in MODE_ORDER:
        reference = reference_modes.get(mode_name)
        stress = stress_modes.get(mode_name)
        if not reference or not stress:
            continue
        if (stress.get("tool_choice_step_count") or 0) <= 0 or (stress.get("oracle_tool_choice_step_count") or 0) <= 0:
            continue

        for metric in LOWER_IS_BETTER_METRICS:
            ref_value = reference.get(metric)
            stress_value = stress.get(metric)
            if ref_value is None or stress_value is None:
                continue
            improvement = ref_value - stress_value
            if improvement >= delta:
                finding = {
                    "mode": mode_name,
                    "metric": metric,
                    "direction": "lower_than_reference",
                    "reference_value": ref_value,
                    "stress_value": stress_value,
                    "improvement": improvement,
                }
                if mode_name == PRIMARY_COMPARISON_MODE:
                    primary_findings.append(finding)
                else:
                    sanity_findings.append(finding)
        for metric in HIGHER_IS_BETTER_METRICS:
            ref_value = reference.get(metric)
            stress_value = stress.get(metric)
            if ref_value is None or stress_value is None:
                continue
            improvement = stress_value - ref_value
            if improvement >= delta:
                finding = {
                    "mode": mode_name,
                    "metric": metric,
                    "direction": "higher_than_reference",
                    "reference_value": ref_value,
                    "stress_value": stress_value,
                    "improvement": improvement,
                }
                if mode_name == PRIMARY_COMPARISON_MODE:
                    primary_findings.append(finding)
                else:
                    sanity_findings.append(finding)

    return primary_findings, sanity_findings


def _build_negative_inversion(
    *,
    stress_modes: dict[str, dict[str, float | None]],
    delta: float,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    selector = stress_modes.get("selector_penalty")
    voi = stress_modes.get("voi_memory_v1")
    if not selector or not voi:
        return findings

    for metric in LOWER_IS_BETTER_METRICS:
        selector_value = selector.get(metric)
        voi_value = voi.get(metric)
        if selector_value is None or voi_value is None:
            continue
        deterioration = voi_value - selector_value
        if deterioration >= delta:
            findings.append(
                {
                    "metric": metric,
                    "direction": "higher_than_selector_penalty",
                    "selector_penalty_value": selector_value,
                    "voi_memory_v1_value": voi_value,
                    "deterioration": deterioration,
                }
            )
    for metric in HIGHER_IS_BETTER_METRICS:
        selector_value = selector.get(metric)
        voi_value = voi.get(metric)
        if selector_value is None or voi_value is None:
            continue
        deterioration = selector_value - voi_value
        if deterioration >= delta:
            findings.append(
                {
                    "metric": metric,
                    "direction": "lower_than_selector_penalty",
                    "selector_penalty_value": selector_value,
                    "voi_memory_v1_value": voi_value,
                    "deterioration": deterioration,
                }
            )
    return findings


def build_seed_battery_summary(
    *,
    reference_run_root: Path,
    stress_run_root: Path,
    delta: float = 0.10,
    route_b_next_action: str = "enter_v4_aggressive_stress_design",
) -> dict[str, Any]:
    reference = _load_pack(reference_run_root)
    stress = _load_pack(stress_run_root)
    trigger_good, sanity_check_deltas = _build_trigger_good(
        reference_modes=reference["modes"],
        stress_modes=stress["modes"],
        delta=delta,
    )
    negative_inversion = _build_negative_inversion(
        stress_modes=stress["modes"],
        delta=delta,
    )
    wrong_tool_surface_exposed = [
        {
            "mode": mode_name,
            "reference_value": reference["modes"].get(mode_name, {}).get("tool_family_mismatch_rate_vs_oracle"),
            "stress_value": stress["modes"].get(mode_name, {}).get("tool_family_mismatch_rate_vs_oracle"),
        }
        for mode_name in MODE_ORDER
        if (
            stress["modes"].get(mode_name, {}).get("tool_family_mismatch_rate_vs_oracle") is not None
            and float(stress["modes"][mode_name]["tool_family_mismatch_rate_vs_oracle"]) > 0.0
        )
    ]
    trigger_good_primary_count = len(trigger_good)
    negative_inversion_count = len(negative_inversion)
    seed11_trigger = bool(
        reference["receipt_ok"]
        and stress["receipt_ok"]
        and (stress["modes"].get(PRIMARY_COMPARISON_MODE, {}).get("tool_choice_step_count") or 0) > 0
        and (stress["modes"].get(PRIMARY_COMPARISON_MODE, {}).get("oracle_tool_choice_step_count") or 0) > 0
        and trigger_good_primary_count >= 2
        and negative_inversion_count == 0
    )
    if negative_inversion:
        route = "C"
        next_action = "ask_gate_memory_selector_diagnosis"
    elif seed11_trigger:
        route = "A"
        next_action = "seed13_mixed_battery_larger_confirmation"
    else:
        route = "B"
        next_action = route_b_next_action

    return {
        "delta": delta,
        "reference": reference,
        "stress": stress,
        "trigger_good": trigger_good,
        "trigger_good_primary_count": trigger_good_primary_count,
        "trigger_good_primary_mode": PRIMARY_COMPARISON_MODE,
        "sanity_check_deltas": sanity_check_deltas,
        "wrong_tool_surface_exposed": wrong_tool_surface_exposed,
        "negative_inversion": negative_inversion,
        "negative_inversion_count": negative_inversion_count,
        "route": route,
        "seed11_trigger": seed11_trigger,
        "next_action": next_action,
    }


def render_seed_battery_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Battery Summary",
        "",
        f"- delta: {summary['delta']:.2f}",
        f"- route: {summary['route']}",
        f"- seed11_trigger: {'true' if summary['seed11_trigger'] else 'false'}",
        f"- next_action: {summary['next_action']}",
        "",
        "## Receipt Layer",
        "",
        f"- reference.receipt_ok: {'true' if summary['reference']['receipt_ok'] else 'false'}",
        f"- stress.receipt_ok: {'true' if summary['stress']['receipt_ok'] else 'false'}",
        "",
        "## Trigger Good",
        "",
    ]
    if summary["trigger_good"]:
        for item in summary["trigger_good"]:
            lines.append(
                f"- {item['mode']} / {item['metric']} / {item['direction']} / "
                f"reference={item['reference_value']:.4f} / stress={item['stress_value']:.4f} / "
                f"delta={item['improvement']:.4f}"
            )
    else:
        lines.append("- none")
    lines.append(f"- trigger_good_primary_mode: {summary['trigger_good_primary_mode']}")
    lines.append(f"- trigger_good_primary_count: {summary['trigger_good_primary_count']}")

    lines.extend(["", "## Sanity Check Deltas", ""])
    if summary["sanity_check_deltas"]:
        for item in summary["sanity_check_deltas"]:
            lines.append(
                f"- {item['mode']} / {item['metric']} / {item['direction']} / "
                f"reference={item['reference_value']:.4f} / stress={item['stress_value']:.4f} / "
                f"delta={item['improvement']:.4f}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Wrong Tool Surface Exposed", ""])
    if summary["wrong_tool_surface_exposed"]:
        for item in summary["wrong_tool_surface_exposed"]:
            lines.append(
                f"- {item['mode']} / reference={item['reference_value']:.4f} / stress={item['stress_value']:.4f}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Negative Inversion", ""])
    if summary["negative_inversion"]:
        for item in summary["negative_inversion"]:
            lines.append(
                f"- {item['metric']} / {item['direction']} / "
                f"selector_penalty={item['selector_penalty_value']:.4f} / "
                f"voi_memory_v1={item['voi_memory_v1_value']:.4f} / "
                f"delta={item['deterioration']:.4f}"
            )
    else:
        lines.append("- none")
    lines.append(f"- negative_inversion_count: {summary['negative_inversion_count']}")

    for pack_name in ("reference", "stress"):
        pack = summary[pack_name]
        lines.extend(["", f"## {pack_name.title()} Pack", ""])
        lines.append(f"- run_name: {pack['run_name']}")
        lines.append(f"- run_root: {pack['run_root']}")
        blocked = pack["blocked"]
        lines.append(
            f"- blocked_step: task_pressure={blocked['task_pressure']}, "
            f"surfaced_ask_deficit={blocked['surfaced_ask_deficit']}, "
            f"hard_cutoff_candidate={blocked['hard_cutoff_candidate']}, "
            f"route_target={blocked['route_target']}"
        )
        lines.append("")
        lines.append("| mode | tool_choice_step_count | oracle_tool_choice_step_count | realized_tool_choice_coverage | choose_swallow | selected_vs_oracle_family_mismatch | tool_family_mismatch | verify_usage_rate | wrong_tool_rate |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mode_name in MODE_ORDER:
            mode = pack["modes"].get(mode_name)
            if not mode:
                continue
            lines.append(
                "| {mode} | {tool} | {oracle} | {coverage:.4f} | {swallow:.4f} | {family:.4f} | {tool_family:.4f} | {verify:.4f} | {wrong:.4f} |".format(
                    mode=mode_name,
                    tool=mode.get("tool_choice_step_count") or 0,
                    oracle=mode.get("oracle_tool_choice_step_count") or 0,
                    coverage=mode.get("realized_tool_choice_coverage") or 0.0,
                    swallow=mode.get("choose_swallow_rate_given_oracle_tool_choice") or 0.0,
                    family=mode.get("selected_vs_oracle_family_mismatch_rate") or 0.0,
                    tool_family=mode.get("tool_family_mismatch_rate_vs_oracle") or 0.0,
                    verify=mode.get("verify_usage_rate") or 0.0,
                    wrong=mode.get("wrong_tool_rate") or 0.0,
                )
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a seed-level dual-battery summary.")
    parser.add_argument("--reference-run-root", type=Path, required=True)
    parser.add_argument("--stress-run-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument(
        "--route-b-next-action",
        default="enter_v4_aggressive_stress_design",
        help="Next action to write when the battery routes to B.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_seed_battery_summary(
        reference_run_root=args.reference_run_root,
        stress_run_root=args.stress_run_root,
        delta=args.delta,
        route_b_next_action=args.route_b_next_action,
    )
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_seed_battery_summary(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
