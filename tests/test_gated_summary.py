from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from eval.gated_summary import render_gated_summary


SCRATCH_ROOT = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime")


@contextlib.contextmanager
def _scratch_dir(prefix: str):
    path = SCRATCH_ROOT / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _control_fixture(control_dir: Path) -> None:
    _write_text(
        control_dir / "CURRENT_CLAIM_BOUNDARY.md",
        "# Current Claim Boundary\n\n## Accepted\n- single-tool ask result\n\n## Strengthened But Not Yet Accepted\n- live family alignment separation\n\n## Not Accepted\n- tool-choice improvement\n\n## Current Primary Bottleneck\n- candidate_coverage\n\n## Current Secondary Bottleneck\n- runtime_backend\n\n## Evidence Level Notes\n- seed-7 is E2\n",
    )
    _write_text(
        control_dir / "PRE_RUN_CARD.md",
        "# Pre-Run Card\n\n- run_name: pre_selector_pruning_decomposition_and_gate_surface_audit_20260416\n- lane: discovery\n- current_focus: pre_selector_pruning_audit\n- execution_lane: discovery\n",
    )
    _write_text(
        control_dir / "POST_RUN_RECEIPT.md",
        "# Post-Run Receipt\n\n- run_name: toolized_verify_necessary_50_liveapi_seed7_20260415\n- run_completed: true\n- jsonl_integrity: pass\n- task_id_pairing: pass\n- missing_tasks: 0\n- corrupted_lines: 0\n- reports_generated: diagnosis.json, diagnosis_report.md\n- safe_for_interpretation: true\n",
    )
    _write_text(
        control_dir / "DECISION_GATE.md",
        "# Decision Gate\n\n## Chosen Action\n- pre_selector_pruning_decomposition_and_gate_surface_audit\n\n## Evidence Used\n- verify loss is now concentrated at the pre-selector filter under voi_memory_v1 and needs gate-facing attribution\n\n## Why Not Action A\n- memory ablation is premature\n\n## Why Not Action B\n- selector change is premature\n\n## Exit Criterion\n- produce pre-selector pruning attribution with verify-vs-scan control and direct gate evidence coverage\n\n## Evidence Level Impact\n- improves E2 diagnosis resolution\n",
    )
    _write_text(
        control_dir / "SHADOW_PROPOSAL.md",
        "# Shadow Proposal\n\n- alternative_action: revisit harder verify_necessary pressure\n- reason_not_chosen_now: candidate mismatch still unresolved\n- signal_that_would_promote_it: diagnostics fail to explain mismatch rows\n",
    )


class GatedSummaryTests(unittest.TestCase):
    def test_required_section_order_matches_cli_summary_format(self) -> None:
        with _scratch_dir("gated_summary_order") as root:
            control_dir = root / "project_control"
            run_root = root / "run"
            _control_fixture(control_dir)
            _write_text(
                run_root / "diagnosis.json",
                json.dumps(
                    {
                        "routing": {
                            "most_likely_layer": "candidate_coverage",
                            "selected_vs_oracle_mismatch_count": 174,
                            "oracle_tool_row_count": 228,
                            "candidate_related_share": 0.9885,
                            "policy_family_error_share": 0.0,
                            "key_symptoms": ["candidate mismatch dominates"],
                            "key_metrics": [{"name": "candidate_related_share", "value": 0.75}],
                            "alternative_explanations": ["runtime_backend"],
                        },
                        "modes": {
                            "full": {"overall": {"verify_candidate_exposed_rate": 1.0}},
                            "selector_penalty": {"overall": {"verify_candidate_exposed_rate": 1.0}},
                            "voi_memory_v1": {
                                "overall": {
                                    "verify_candidate_exposed_rate": 0.2308,
                                    "best_verify_missing_reason_generated_but_pruned_rate": 0.7692,
                                }
                            },
                        },
                        "pre_selector_pruning": {
                            "voi_memory_v1": {
                                "pre_selector_applicable_count": 115,
                                "pre_selector_retained_count": 27,
                                "pre_selector_retained_rate": 0.2348,
                                "filter_rule_count": 88,
                                "filter_rule_without_direct_gate_evidence": 0,
                            }
                        },
                        "blocked_step_audit": {
                            "voi_memory_v1": {
                                "blocked_step_count": 50,
                                "route_target": "task_pressure",
                                "direct_choose_correct_rate": 0.96,
                            }
                        },
                    }
                ),
            )
            summary, meta = render_gated_summary(
                control_dir=control_dir,
                run_root=run_root,
                action_type="implementation_change",
                action_target="candidate_shape_normalization_and_exposure",
                lane="discovery",
                owner="Codex",
                run_context="candidate_shape_normalization_and_exposure_20260415",
            )
            ordered_sections = [
                "[CLAIM BOUNDARY]",
                "[LAST RUN RECEIPT]",
                "[DIAGNOSIS SNAPSHOT]",
                "[DECISION GATE]",
                "[SHADOW PROPOSAL]",
                "[GATE STATUS]",
                "[NEXT STEP]",
            ]
            indices = [summary.index(section) for section in ordered_sections]
            self.assertEqual(indices, sorted(indices))
            self.assertEqual("Header", meta["required_section_order"][0])

    def test_missing_control_files_block_execution(self) -> None:
        with _scratch_dir("gated_summary_block") as root:
            control_dir = root / "project_control"
            run_root = root / "run"
            control_dir.mkdir(parents=True, exist_ok=True)
            _write_text(run_root / "diagnosis.json", json.dumps({"routing": {}}))
            _, meta = render_gated_summary(
                control_dir=control_dir,
                run_root=run_root,
                action_type="implementation_change",
                action_target="candidate_shape_normalization_and_exposure",
                lane="discovery",
                owner="Codex",
                run_context="candidate_shape_normalization_and_exposure_20260415",
            )
            self.assertFalse(meta["execution_allowed"])
            self.assertIn("missing_control_files", meta["block_reason"])

    def test_complete_discovery_summary_allows_execution(self) -> None:
        with _scratch_dir("gated_summary_ready") as root:
            control_dir = root / "project_control"
            run_root = root / "run"
            _control_fixture(control_dir)
            _write_text(
                run_root / "diagnosis.json",
                json.dumps(
                    {
                        "routing": {
                            "most_likely_layer": "candidate_coverage",
                            "selected_vs_oracle_mismatch_count": 174,
                            "oracle_tool_row_count": 228,
                            "candidate_related_share": 0.9885,
                            "policy_family_error_share": 0.0,
                            "key_symptoms": ["candidate mismatch dominates"],
                            "key_metrics": [{"name": "candidate_related_share", "value": 0.75}],
                            "alternative_explanations": ["runtime_backend"],
                        },
                        "modes": {
                            "full": {"overall": {"verify_candidate_exposed_rate": 1.0}},
                            "selector_penalty": {"overall": {"verify_candidate_exposed_rate": 1.0}},
                            "voi_memory_v1": {
                                "overall": {
                                    "verify_candidate_exposed_rate": 0.2308,
                                    "best_verify_missing_reason_generated_but_pruned_rate": 0.7692,
                                }
                            },
                        },
                        "pre_selector_pruning": {
                            "voi_memory_v1": {
                                "pre_selector_applicable_count": 115,
                                "pre_selector_retained_count": 27,
                                "pre_selector_retained_rate": 0.2348,
                                "filter_rule_count": 88,
                                "filter_rule_without_direct_gate_evidence": 0,
                            }
                        },
                        "blocked_step_audit": {
                            "voi_memory_v1": {
                                "blocked_step_count": 50,
                                "route_target": "task_pressure",
                                "direct_choose_correct_rate": 0.96,
                            }
                        },
                    }
                ),
            )
            summary, meta = render_gated_summary(
                control_dir=control_dir,
                run_root=run_root,
                action_type="implementation_change",
                action_target="candidate_shape_normalization_and_exposure",
                lane="discovery",
                owner="Codex",
                run_context="candidate_shape_normalization_and_exposure_20260415",
            )
            self.assertTrue(meta["execution_allowed"])
            self.assertIn("execution_allowed: true", summary)
            self.assertIn("action_type: implementation_change", summary)
            self.assertIn("selected_vs_oracle_mismatch_count: 174", summary)
            self.assertIn("candidate_related_share: 0.9885", summary)
            self.assertIn("most_likely_layer: task_pressure", summary)
            self.assertIn("current_focus: pre_selector_pruning_audit", summary)
            self.assertIn("execution_lane: discovery", summary)
            self.assertIn("voi_memory_v1_generated_but_pruned_rate: 0.7692", summary)
            self.assertIn("voi_memory_v1_pre_selector_applicable_count: 115", summary)
            self.assertIn("voi_memory_v1_pre_selector_retained_count: 27", summary)
            self.assertIn("voi_memory_v1_pre_selector_retained_rate: 0.2348", summary)
            self.assertIn("voi_memory_v1_filter_rule_count: 88", summary)
            self.assertIn("voi_memory_v1_filter_rule_without_direct_gate_evidence: 0", summary)
            self.assertIn("blocked_step_count: 50", summary)
            self.assertIn("blocked_step_route_target: task_pressure", summary)
            self.assertIn("blocked_step_direct_choose_correct_rate: 0.96", summary)


if __name__ == "__main__":
    unittest.main()
