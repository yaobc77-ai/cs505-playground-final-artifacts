from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECTION_ORDER = [
    "Header",
    "Claim Boundary",
    "Last Run Receipt",
    "Diagnosis Snapshot",
    "Decision Gate",
    "Shadow Proposal",
    "Gate Status",
    "Next Step Execution Notice",
]


def _parse_key_value_markdown(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        key, value = line[2:].split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _parse_claim_boundary(path: Path) -> dict[str, list[str]]:
    mapping = {
        "Accepted": "accepted",
        "Strengthened But Not Yet Accepted": "strengthened_not_accepted",
        "Not Accepted": "not_accepted",
        "Current Primary Bottleneck": "primary_bottleneck",
        "Current Secondary Bottleneck": "secondary_bottleneck",
        "Evidence Level Notes": "evidence_level_notes",
    }
    result = {value: [] for value in mapping.values()}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current_key = mapping.get(line[3:].strip())
            continue
        stripped = line.strip()
        if current_key and stripped.startswith("- "):
            result[current_key].append(stripped[2:].strip())
    return result


def _parse_sectioned_markdown(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            if current_key is not None:
                result[current_key] = " ".join(current_lines).strip()
            current_key = line[3:].strip()
            current_lines = []
            continue
        stripped = line.strip()
        if current_key and stripped.startswith("- "):
            current_lines.append(stripped[2:].strip())
    if current_key is not None:
        result[current_key] = " ".join(current_lines).strip()
    return result


def _load_diagnosis_snapshot(path: Path) -> dict[str, Any]:
    diagnosis = json.loads(path.read_text(encoding="utf-8"))
    routing = diagnosis.get("routing") or {}
    modes = diagnosis.get("modes") or {}
    pre_selector = (diagnosis.get("pre_selector_pruning") or {}).get("voi_memory_v1") or {}
    blocked_step_audit = (diagnosis.get("blocked_step_audit") or {}).get("voi_memory_v1") or {}
    full_overall = ((modes.get("full") or {}).get("overall") or {})
    selector_overall = ((modes.get("selector_penalty") or {}).get("overall") or {})
    voi_overall = ((modes.get("voi_memory_v1") or {}).get("overall") or {})
    most_likely_layer = routing.get("most_likely_layer", "unknown")
    blocked_step_route_target = blocked_step_audit.get("route_target")
    if blocked_step_audit.get("blocked_step_count") and blocked_step_route_target and blocked_step_route_target != "mixed":
        most_likely_layer = blocked_step_route_target
    return {
        "most_likely_layer": most_likely_layer,
        "selected_vs_oracle_mismatch_count": routing.get("selected_vs_oracle_mismatch_count"),
        "oracle_tool_row_count": routing.get("oracle_tool_row_count"),
        "candidate_related_share": routing.get("candidate_related_share"),
        "policy_family_error_share": routing.get("policy_family_error_share"),
        "full_verify_exposed_rate": full_overall.get("verify_candidate_exposed_rate"),
        "selector_penalty_verify_exposed_rate": selector_overall.get("verify_candidate_exposed_rate"),
        "voi_memory_v1_verify_exposed_rate": voi_overall.get("verify_candidate_exposed_rate"),
        "voi_memory_v1_generated_but_pruned_rate": voi_overall.get(
            "best_verify_missing_reason_generated_but_pruned_rate"
        ),
        "voi_memory_v1_pre_selector_applicable_count": pre_selector.get(
            "pre_selector_applicable_count"
        ),
        "voi_memory_v1_pre_selector_retained_count": pre_selector.get(
            "pre_selector_retained_count"
        ),
        "voi_memory_v1_pre_selector_retained_rate": pre_selector.get(
            "pre_selector_retained_rate"
        ),
        "voi_memory_v1_filter_rule_count": pre_selector.get("filter_rule_count"),
        "voi_memory_v1_filter_rule_without_direct_gate_evidence": pre_selector.get(
            "filter_rule_without_direct_gate_evidence"
        ),
        "blocked_step_count": blocked_step_audit.get("blocked_step_count"),
        "blocked_step_route_target": blocked_step_route_target,
        "blocked_step_direct_choose_correct_rate": blocked_step_audit.get("direct_choose_correct_rate"),
        "key_symptoms": list(routing.get("key_symptoms") or []),
        "key_metrics": list(routing.get("key_metrics") or []),
        "alternative_explanations": list(routing.get("alternative_explanations") or []),
    }


def _format_list_block(values: list[str], *, fallback: str = "none") -> list[str]:
    if not values:
        return [f"  - {fallback}"]
    return [f"  - {value}" for value in values]


def _required_control_paths(control_dir: Path) -> dict[str, Path]:
    return {
        "claim_boundary": control_dir / "CURRENT_CLAIM_BOUNDARY.md",
        "pre_run_card": control_dir / "PRE_RUN_CARD.md",
        "post_run_receipt": control_dir / "POST_RUN_RECEIPT.md",
        "decision_gate": control_dir / "DECISION_GATE.md",
        "shadow_proposal": control_dir / "SHADOW_PROPOSAL.md",
    }


def render_gated_summary(
    *,
    control_dir: Path,
    run_root: Path,
    action_type: str,
    action_target: str,
    lane: str,
    owner: str,
    run_context: str,
) -> tuple[str, dict[str, Any]]:
    required_paths = _required_control_paths(control_dir)
    missing = [
        name
        for name, path in required_paths.items()
        if not path.exists()
    ]
    diagnosis_path = run_root / "diagnosis.json"

    pre_run_card_present = required_paths["pre_run_card"].exists()
    post_run_receipt_present = required_paths["post_run_receipt"].exists()
    decision_gate_present = required_paths["decision_gate"].exists()
    shadow_proposal_present = required_paths["shadow_proposal"].exists()
    diagnosis_present = diagnosis_path.exists()

    claim_boundary = (
        _parse_claim_boundary(required_paths["claim_boundary"])
        if required_paths["claim_boundary"].exists()
        else {
            "accepted": [],
            "strengthened_not_accepted": [],
            "not_accepted": [],
            "primary_bottleneck": [],
            "secondary_bottleneck": [],
            "evidence_level_notes": [],
        }
    )
    receipt = _parse_key_value_markdown(required_paths["post_run_receipt"]) if post_run_receipt_present else {}
    pre_run_card = _parse_key_value_markdown(required_paths["pre_run_card"]) if pre_run_card_present else {}
    decision_gate = _parse_sectioned_markdown(required_paths["decision_gate"]) if decision_gate_present else {}
    shadow_proposal = _parse_key_value_markdown(required_paths["shadow_proposal"]) if shadow_proposal_present else {}
    diagnosis_snapshot = _load_diagnosis_snapshot(diagnosis_path) if diagnosis_present else {
        "most_likely_layer": "unknown",
        "key_symptoms": [],
        "key_metrics": [],
        "alternative_explanations": [],
    }

    jsonl_integrity = receipt.get("jsonl_integrity", "fail")
    task_id_pairing = receipt.get("task_id_pairing", "fail")
    block_reason = "none"
    if missing:
        block_reason = f"missing_control_files:{','.join(missing)}"
    elif not diagnosis_present:
        block_reason = "missing_diagnosis"
    elif jsonl_integrity != "pass":
        block_reason = "jsonl_integrity_failed"
    elif task_id_pairing != "pass":
        block_reason = "task_id_pairing_failed"

    execution_allowed = block_reason == "none"

    key_metrics_lines: list[str] = []
    for metric in diagnosis_snapshot["key_metrics"]:
        if isinstance(metric, dict):
            key_metrics_lines.append(f"  - name: {metric.get('name')}")
            key_metrics_lines.append(f"    value: {metric.get('value')}")
        else:
            key_metrics_lines.append("  - name: metric")
            key_metrics_lines.append(f"    value: {metric}")
    if not key_metrics_lines:
        key_metrics_lines = ["  - name: none", "    value: unavailable"]

    summary = [
        "=== GATED DECISION SUMMARY ===",
        f"run_context: {run_context}",
        f"date: {datetime.now(timezone.utc).isoformat()}",
        f"owner: {owner}",
        f"lane: {lane}",
        "",
        "[CLAIM BOUNDARY]",
        "accepted:",
        *_format_list_block(claim_boundary["accepted"]),
        "strengthened_not_accepted:",
        *_format_list_block(claim_boundary["strengthened_not_accepted"]),
        "not_accepted:",
        *_format_list_block(claim_boundary["not_accepted"]),
        "primary_bottleneck:",
        *_format_list_block(claim_boundary["primary_bottleneck"]),
        "secondary_bottleneck:",
        *_format_list_block(claim_boundary["secondary_bottleneck"]),
        f"current_focus: {pre_run_card.get('current_focus', 'missing')}",
        f"execution_lane: {pre_run_card.get('execution_lane', lane)}",
        "",
        "[LAST RUN RECEIPT]",
        f"run_name: {receipt.get('run_name', 'missing')}",
        f"run_completed: {receipt.get('run_completed', 'false')}",
        f"jsonl_integrity: {jsonl_integrity}",
        f"task_id_pairing: {task_id_pairing}",
        f"missing_tasks: {receipt.get('missing_tasks', 'missing')}",
        f"corrupted_lines: {receipt.get('corrupted_lines', 'missing')}",
        "reports_generated:",
        *_format_list_block(
            [item.strip() for item in str(receipt.get("reports_generated", "")).split(",") if item.strip()]
        ),
        f"safe_for_interpretation: {receipt.get('safe_for_interpretation', 'false')}",
        "",
        "[DIAGNOSIS SNAPSHOT]",
        f"most_likely_layer: {diagnosis_snapshot['most_likely_layer']}",
        f"selected_vs_oracle_mismatch_count: {diagnosis_snapshot.get('selected_vs_oracle_mismatch_count', 'missing')}",
        f"oracle_tool_row_count: {diagnosis_snapshot.get('oracle_tool_row_count', 'missing')}",
        f"candidate_related_share: {diagnosis_snapshot.get('candidate_related_share', 'missing')}",
        f"policy_family_error_share: {diagnosis_snapshot.get('policy_family_error_share', 'missing')}",
        f"full_verify_exposed_rate: {diagnosis_snapshot.get('full_verify_exposed_rate', 'missing')}",
        f"selector_penalty_verify_exposed_rate: {diagnosis_snapshot.get('selector_penalty_verify_exposed_rate', 'missing')}",
        f"voi_memory_v1_verify_exposed_rate: {diagnosis_snapshot.get('voi_memory_v1_verify_exposed_rate', 'missing')}",
        f"voi_memory_v1_generated_but_pruned_rate: {diagnosis_snapshot.get('voi_memory_v1_generated_but_pruned_rate', 'missing')}",
        f"voi_memory_v1_pre_selector_applicable_count: {diagnosis_snapshot.get('voi_memory_v1_pre_selector_applicable_count', 'missing')}",
        f"voi_memory_v1_pre_selector_retained_count: {diagnosis_snapshot.get('voi_memory_v1_pre_selector_retained_count', 'missing')}",
        f"voi_memory_v1_pre_selector_retained_rate: {diagnosis_snapshot.get('voi_memory_v1_pre_selector_retained_rate', 'missing')}",
        f"voi_memory_v1_filter_rule_count: {diagnosis_snapshot.get('voi_memory_v1_filter_rule_count', 'missing')}",
        "voi_memory_v1_filter_rule_without_direct_gate_evidence: "
        f"{diagnosis_snapshot.get('voi_memory_v1_filter_rule_without_direct_gate_evidence', 'missing')}",
        f"blocked_step_count: {diagnosis_snapshot.get('blocked_step_count', 'missing')}",
        f"blocked_step_route_target: {diagnosis_snapshot.get('blocked_step_route_target', 'missing')}",
        "blocked_step_direct_choose_correct_rate: "
        f"{diagnosis_snapshot.get('blocked_step_direct_choose_correct_rate', 'missing')}",
        "key_symptoms:",
        *_format_list_block(diagnosis_snapshot["key_symptoms"]),
        "key_metrics:",
        *key_metrics_lines,
        "alternative_explanations:",
        *_format_list_block(diagnosis_snapshot["alternative_explanations"]),
        "",
        "[DECISION GATE]",
        f"chosen_action: {decision_gate.get('Chosen Action', 'missing')}",
        f"why_this_action: {decision_gate.get('Evidence Used', 'missing')}",
        f"why_not_action_a: {decision_gate.get('Why Not Action A', 'missing')}",
        f"why_not_action_b: {decision_gate.get('Why Not Action B', 'missing')}",
        f"exit_criterion: {decision_gate.get('Exit Criterion', 'missing')}",
        f"evidence_level_impact: {decision_gate.get('Evidence Level Impact', 'missing')}",
        "",
        "[SHADOW PROPOSAL]",
        f"alternative_action: {shadow_proposal.get('alternative_action', 'missing')}",
        f"reason_not_chosen_now: {shadow_proposal.get('reason_not_chosen_now', 'missing')}",
        f"signal_that_would_promote_it: {shadow_proposal.get('signal_that_would_promote_it', 'missing')}",
        "",
        "[GATE STATUS]",
        f"pre_run_card_present: {'true' if pre_run_card_present else 'false'}",
        f"post_run_receipt_present: {'true' if post_run_receipt_present else 'false'}",
        f"diagnosis_present: {'true' if diagnosis_present else 'false'}",
        f"decision_gate_present: {'true' if decision_gate_present else 'false'}",
        f"shadow_proposal_present: {'true' if shadow_proposal_present else 'false'}",
        f"execution_allowed: {'true' if execution_allowed else 'false'}",
        f"block_reason: {block_reason}",
        "",
        "[NEXT STEP]",
        f"action_type: {action_type}",
        f"action_target: {action_target}",
        f"status: {'ready' if execution_allowed else 'blocked'}",
        "",
        "GATE_JSON="
        + json.dumps(
            {
                "lane": lane,
                "most_likely_layer": diagnosis_snapshot["most_likely_layer"],
                "execution_allowed": execution_allowed,
                "block_reason": block_reason,
                "chosen_action": decision_gate.get("Chosen Action", "missing"),
            }
        ),
    ]
    return "\n".join(summary), {
        "pre_run_card_present": pre_run_card_present,
        "post_run_receipt_present": post_run_receipt_present,
        "diagnosis_present": diagnosis_present,
        "decision_gate_present": decision_gate_present,
        "shadow_proposal_present": shadow_proposal_present,
        "execution_allowed": execution_allowed,
        "block_reason": block_reason,
        "required_section_order": list(SECTION_ORDER),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a gated execution summary from project control artifacts.")
    parser.add_argument("--control-dir", type=Path, default=Path("project_control"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--action-type", type=str, required=True)
    parser.add_argument("--action-target", type=str, required=True)
    parser.add_argument("--lane", type=str, default="discovery")
    parser.add_argument("--owner", type=str, default="Codex")
    parser.add_argument("--run-context", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _ = render_gated_summary(
        control_dir=args.control_dir,
        run_root=args.run_root,
        action_type=args.action_type,
        action_target=args.action_target,
        lane=args.lane,
        owner=args.owner,
        run_context=args.run_context or args.action_target,
    )
    print(summary)


if __name__ == "__main__":
    main()
