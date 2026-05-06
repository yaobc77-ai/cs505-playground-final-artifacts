from __future__ import annotations

import unittest

from eval.diagnose_reflection import (
    _analyze_step_record,
    _best_surface_eval_for_action_type,
    _classify_blocked_step,
    _classify_pre_selector_block,
    _collect_blocked_step_audit_rows,
    _collect_pre_selector_pruning_rows,
    _classify_oracle_mismatch,
    _group_step_rows,
    _render_blocked_step_audit,
    _render_pre_selector_pruning_audit,
    _render_diagnosis_report,
    _summarize_blocked_step_audit,
    _summarize_pre_selector_pruning,
    _surface_shape_from_candidate,
    _summarize_step_rows,
)


def _tool_step_record(
    *,
    action: str,
    scan_eval: float,
    verify_eval: float,
    choose_eval: float = 0.6,
) -> dict[str, object]:
    if action.startswith("ASK_SCAN_"):
        selected_eval = scan_eval
    elif action.startswith("ASK_VERIFY_"):
        selected_eval = verify_eval
    else:
        selected_eval = choose_eval
    return {
        "task_id": "task-0001",
        "step": 1,
        "action": action,
        "task_tags": [
            "difficulty:easy",
            "noise:0.05",
            "mode:voi_memory_v1",
            "tool_profile:verify_necessary",
            "tool_subtype:scan_then_verify",
        ],
        "belief": {
            "margin": 0.2,
            "asked_count": 0,
        },
        "self_model": {
            "info_seeking": 0.7,
            "cost_sensitivity": 0.4,
            "confidence_threshold": 1.25,
        },
        "candidates": [
            {"name": "scan", "actions": ["ASK_SCAN_1", "CHOOSE_BEST"]},
            {"name": "verify", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"]},
            {"name": "choose", "actions": ["CHOOSE_A"]},
        ],
        "evaluations": [
            {"candidate_name": "scan", "score": scan_eval},
            {"candidate_name": "verify", "score": verify_eval},
            {"candidate_name": "choose", "score": choose_eval},
        ],
        "retrieved_memory_ids": [],
        "selection_breakdown": {"eval_score": selected_eval},
        "total_selection_score": selected_eval,
        "ask_gate_allowed": True,
        "ask_gate_score": 0.9,
        "ask_gate_reason": "strong_positive_voi",
        "retrieved_decision_lessons": {
            "positive": [],
            "caution": [],
            "summary": {
                "helpful_ask": 0.0,
                "choose_regret": 0.0,
                "regretful_ask": 0.0,
                "choose_helpful": 0.0,
            },
        },
        "written_decision_lesson": None,
    }


def _oracle_gap_task() -> dict[str, object]:
    return {
        "task_id": "task-0001",
        "answer": "A",
        "difficulty": "easy",
        "noise_level": 0.05,
        "ask_cost": -0.05,
        "max_steps": 4,
        "toolized": True,
        "clues": [
            {
                "id": 1,
                "scan_scores": {"A": 0, "B": 2, "C": 0},
                "verify_scores": {"A": 0, "B": 3, "C": 0},
                "scan_reliability": 0.4,
                "verify_reliability": 0.6,
                "scan_cost": -0.02,
                "verify_cost": -0.08,
                "scan_text": "scan 1",
                "verify_text": "verify 1",
            },
            {
                "id": 2,
                "scan_scores": {"A": 0, "B": 1, "C": 0},
                "verify_scores": {"A": 4, "B": -1, "C": 0},
                "scan_reliability": 0.9,
                "verify_reliability": 0.95,
                "scan_cost": -0.02,
                "verify_cost": -0.08,
                "scan_text": "scan 2",
                "verify_text": "verify 2",
            },
        ],
    }


def _scan_oracle_task() -> dict[str, object]:
    return {
        "task_id": "task-0001",
        "answer": "A",
        "difficulty": "easy",
        "noise_level": 0.05,
        "ask_cost": -0.05,
        "max_steps": 4,
        "toolized": True,
        "clues": [
            {
                "id": 1,
                "scan_scores": {"A": 4, "B": 0, "C": 0},
                "verify_scores": {"A": 1, "B": 0, "C": 0},
                "scan_reliability": 1.0,
                "verify_reliability": 0.8,
                "scan_cost": -0.03,
                "verify_cost": -0.08,
                "scan_text": "scan 1",
                "verify_text": "verify 1",
            },
            {
                "id": 2,
                "scan_scores": {"A": 2, "B": 0, "C": 0},
                "verify_scores": {"A": 0, "B": 2, "C": 0},
                "scan_reliability": 0.9,
                "verify_reliability": 0.9,
                "scan_cost": -0.03,
                "verify_cost": -0.08,
                "scan_text": "scan 2",
                "verify_text": "verify 2",
            },
        ],
    }


def _pre_selector_step_row(
    *,
    ask_gate_reason: str,
    ask_gate_allowed: bool,
    pre_selector_has_direct_evidence: bool,
    pre_selector_filter_applied: bool,
    pre_selector_filtered_pairs_empty: bool,
    verify_entered_selector: bool,
    tool_subtype: str = "verify_first",
) -> dict[str, object]:
    return {
        "seed": 7,
        "mode": "voi_memory_v1",
        "task_id": "task-0001",
        "task_index": 1,
        "step_index": 1,
        "difficulty": "easy",
        "noise_level": "0.05",
        "tool_profile": "verify_necessary",
        "tool_subtype": tool_subtype,
        "ask_gate_enabled": True,
        "ask_gate_allowed": ask_gate_allowed,
        "ask_gate_reason": ask_gate_reason,
        "ask_gate_score": -0.4,
        "ask_advantage": -0.12,
        "ask_gate_best_ask_family": "verify",
        "ask_gate_best_ask_eval": 0.31,
        "ask_gate_best_choose_eval": 0.55,
        "ask_gate_voi_ratio": -1.5,
        "ask_gate_uncertainty": 0.42,
        "ask_gate_base_uncertainty": 0.35,
        "ask_gate_reliability_uncertainty": 0.1,
        "ask_gate_evidence_conflict": 0.05,
        "ask_gate_memory_signal": -0.2,
        "ask_gate_helpful_ask": 0.0,
        "ask_gate_choose_regret": 0.1,
        "ask_gate_regretful_ask": 0.3,
        "ask_gate_choose_helpful": 0.0,
        "pre_selector_has_direct_evidence": pre_selector_has_direct_evidence,
        "pre_selector_filter_applied": pre_selector_filter_applied,
        "pre_selector_filtered_pairs_empty": pre_selector_filtered_pairs_empty,
        "remaining_verify_actions": ["ASK_VERIFY_1"],
        "remaining_scan_actions": ["ASK_SCAN_1"],
        "final_verify_lineage_keys": ["ASK_VERIFY_1"],
        "selector_verify_lineage_keys": ["ASK_VERIFY_1"] if verify_entered_selector else [],
        "final_scan_lineage_keys": ["ASK_SCAN_1"],
        "selector_scan_lineage_keys": [],
        "verify_lineage_audit": [
            {
                "verify_lineage_key": "ASK_VERIFY_1",
                "verify_pruned_stage": "none" if verify_entered_selector else "pre_selector_filter",
                "verify_pruned_reason": "none" if verify_entered_selector else "filter_rule",
                "verify_retained_before_selector": True,
                "verify_entered_selector": verify_entered_selector,
            }
        ],
    }


def _blocked_audit_step_record(
    *,
    choose_action: str,
    choose_eval: float,
    surfaced_ask_action: str | None = None,
    surfaced_ask_eval: float | None = None,
) -> dict[str, object]:
    candidates = [
        {"name": "choose_now", "actions": [choose_action], "category": "choose_now"},
    ]
    evaluations = [
        {"candidate_name": "choose_now", "score": choose_eval},
    ]
    if surfaced_ask_action is not None and surfaced_ask_eval is not None:
        candidates.append(
            {
                "name": "ask_surface",
                "actions": [surfaced_ask_action, "CHOOSE_BEST"],
                "category": "ask_then_choose",
            }
        )
        evaluations.append({"candidate_name": "ask_surface", "score": surfaced_ask_eval})
    return {
        "task_id": "task-0001",
        "step": 1,
        "action": choose_action,
        "ask_gate_reason": "strong_negative_voi",
        "candidates": candidates,
        "evaluations": evaluations,
        "ask_gate_inputs": {
            "best_choose_eval": choose_eval,
            "best_ask_eval": surfaced_ask_eval,
        },
    }


def _blocked_step_row() -> dict[str, object]:
    return {
        "seed": 7,
        "task_id": "task-0001",
        "task_index": 1,
        "step_index": 1,
        "difficulty": "easy",
        "noise_level": "0.05",
        "tool_profile": "verify_necessary",
        "tool_subtype": "verify_first",
        "selected_type": "CHOOSE",
        "remaining_scan_action_count": 2,
        "remaining_verify_action_count": 2,
    }


class DiagnoseReflectionToolMetricsTests(unittest.TestCase):
    def test_analyze_step_record_marks_wrong_scan_as_under_tool(self) -> None:
        row = _analyze_step_record(
            seed=7,
            mode_name="voi_memory_v1",
            step_record=_tool_step_record(
                action="ASK_SCAN_1",
                scan_eval=0.90,
                verify_eval=0.97,
            ),
            memory_index={},
        )
        self.assertEqual("scan", row["selected_tool_family"])
        self.assertTrue(row["tool_choice_applicable"])
        self.assertEqual("verify", row["tool_optimal_family"])
        self.assertTrue(row["wrong_tool_choice"])
        self.assertEqual("verify_necessary", row["tool_profile"])
        self.assertEqual("scan_then_verify", row["tool_subtype"])
        self.assertAlmostEqual(0.07, float(row["under_tool_cost"]), places=4)
        self.assertAlmostEqual(0.0, float(row["over_tool_cost"]), places=4)

    def test_analyze_step_record_reports_oracle_family_coverage_gap(self) -> None:
        step_record = {
            **_tool_step_record(
                action="ASK_VERIFY_1",
                scan_eval=0.40,
                verify_eval=0.55,
                choose_eval=0.0,
            ),
            "belief": {
                "margin": 0.0,
                "asked_count": 0,
                "remaining_scan_actions": ["ASK_SCAN_1", "ASK_SCAN_2"],
                "remaining_verify_actions": ["ASK_VERIFY_1", "ASK_VERIFY_2"],
            },
            "candidates": [
                {"name": "scan", "actions": ["ASK_SCAN_1", "CHOOSE_BEST"]},
                {"name": "verify", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"]},
                {"name": "choose", "actions": ["CHOOSE_A"]},
            ],
            "evaluations": [
                {"candidate_name": "scan", "score": 0.40},
                {"candidate_name": "verify", "score": 0.55},
                {"candidate_name": "choose", "score": 0.0},
            ],
            "template_candidates": [
                {"name": "scan", "actions": ["ASK_SCAN_1", "CHOOSE_BEST"]},
                {"name": "verify", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"]},
                {"name": "choose", "actions": ["CHOOSE_A"]},
            ],
            "selector_candidates": [
                {"name": "scan", "actions": ["ASK_SCAN_1", "CHOOSE_BEST"]},
                {"name": "verify", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"]},
                {"name": "choose", "actions": ["CHOOSE_A"]},
            ],
            "candidate_backend": "template",
        }
        row = _analyze_step_record(
            seed=7,
            mode_name="full",
            step_record=step_record,
            memory_index={},
            task_config=_oracle_gap_task(),
            prior_actions=[],
        )
        self.assertTrue(row["oracle_tool_choice_applicable"])
        self.assertEqual("verify", row["oracle_tool_optimal_family"])
        self.assertEqual("ASK_VERIFY_2", row["oracle_best_verify_action"])
        self.assertFalse(row["final_oracle_verify_action_covered"])
        self.assertGreater(float(row["final_verify_eval_gap"]), 0.0)
        self.assertEqual(2, row["verify_lineage_total_count"])
        self.assertIn("verify_pruned_stage", row["verify_lineage_audit"][0])

    def test_analyze_step_record_marks_choose_swallow_against_oracle_tool_need(self) -> None:
        step_record = {
            **_tool_step_record(
                action="CHOOSE_A",
                scan_eval=0.40,
                verify_eval=0.55,
                choose_eval=0.0,
            ),
            "belief": {
                "margin": 0.0,
                "asked_count": 0,
                "remaining_scan_actions": ["ASK_SCAN_1", "ASK_SCAN_2"],
                "remaining_verify_actions": ["ASK_VERIFY_1", "ASK_VERIFY_2"],
            },
        }
        row = _analyze_step_record(
            seed=7,
            mode_name="full",
            step_record=step_record,
            memory_index={},
            task_config=_oracle_gap_task(),
            prior_actions=[],
        )
        self.assertTrue(row["selected_vs_oracle_family_mismatch"])
        self.assertEqual("choose_swallow", row["selected_vs_oracle_mismatch_kind"])
        self.assertEqual("none", row["tool_family_error_source"])
        self.assertFalse(row["wrong_tool_choice"])

    def test_analyze_step_record_marks_surface_family_distortion_without_wrong_tool(self) -> None:
        step_record = {
            **_tool_step_record(
                action="ASK_VERIFY_1",
                scan_eval=0.40,
                verify_eval=0.55,
                choose_eval=0.0,
            ),
            "belief": {
                "margin": 0.0,
                "asked_count": 0,
                "remaining_scan_actions": ["ASK_SCAN_1", "ASK_SCAN_2"],
                "remaining_verify_actions": ["ASK_VERIFY_1", "ASK_VERIFY_2"],
            },
        }
        row = _analyze_step_record(
            seed=7,
            mode_name="full",
            step_record=step_record,
            memory_index={},
            task_config=_scan_oracle_task(),
            prior_actions=[],
        )
        self.assertTrue(row["selected_vs_oracle_family_mismatch"])
        self.assertEqual("tool_family_mismatch", row["selected_vs_oracle_mismatch_kind"])
        self.assertEqual("surface_family_distortion", row["tool_family_error_source"])
        self.assertFalse(row["wrong_tool_choice"])
        self.assertIn("best_scan_surface_shape_before", row)
        self.assertIn("best_verify_surface_shape_after", row)
        self.assertIn("verify_candidate_exposed", row)
        self.assertIn("best_verify_missing_reason", row)

    def test_analyze_step_record_marks_policy_family_error_when_selection_flips_oracle(self) -> None:
        step_record = {
            **_tool_step_record(
                action="ASK_SCAN_1",
                scan_eval=0.40,
                verify_eval=0.55,
                choose_eval=0.0,
            ),
            "belief": {
                "margin": 0.0,
                "asked_count": 0,
                "remaining_scan_actions": ["ASK_SCAN_1", "ASK_SCAN_2"],
                "remaining_verify_actions": ["ASK_VERIFY_1", "ASK_VERIFY_2"],
            },
        }
        row = _analyze_step_record(
            seed=7,
            mode_name="full",
            step_record=step_record,
            memory_index={},
            task_config=_oracle_gap_task(),
            prior_actions=[],
        )
        self.assertTrue(row["selected_vs_oracle_family_mismatch"])
        self.assertEqual("tool_family_mismatch", row["selected_vs_oracle_mismatch_kind"])
        self.assertEqual("policy_family_error", row["tool_family_error_source"])
        self.assertTrue(row["wrong_tool_choice"])

    def test_oracle_mismatch_classifier_covers_mixed_branch(self) -> None:
        mismatch, kind, source = _classify_oracle_mismatch(
            selected_type="ASK",
            selected_tool_family="clue",
            tool_choice_applicable=True,
            tool_optimal_family="verify",
            oracle_tool_choice_applicable=True,
            oracle_tool_optimal_family="scan",
        )
        self.assertTrue(mismatch)
        self.assertEqual("tool_family_mismatch", kind)
        self.assertEqual("mixed_surface_and_policy_error", source)

    def test_surface_shape_classifier_covers_all_values(self) -> None:
        self.assertEqual(
            "canonical_choose_best",
            _surface_shape_from_candidate({"actions": ["ASK_SCAN_1", "CHOOSE_BEST"], "category": "ask_then_choose"}),
        )
        self.assertEqual(
            "fixed_choose",
            _surface_shape_from_candidate({"actions": ["ASK_VERIFY_1", "CHOOSE_A"], "category": "ask_then_choose_fixed"}),
        )
        self.assertEqual(
            "ask_only",
            _surface_shape_from_candidate({"actions": ["ASK_VERIFY_1"], "category": "ask_only"}),
        )
        self.assertEqual(
            "multi_step",
            _surface_shape_from_candidate({"actions": ["ASK_SCAN_1", "ASK_VERIFY_2"], "category": "ask_only"}),
        )
        self.assertEqual("none", _surface_shape_from_candidate({"actions": ["CHOOSE_A"], "category": "choose_now"}))

    def test_group_step_rows_supports_tool_profile_and_subtype(self) -> None:
        rows = [
            _analyze_step_record(
                seed=7,
                mode_name="voi_memory_v1",
                step_record=_tool_step_record(
                    action="ASK_SCAN_1",
                    scan_eval=0.90,
                    verify_eval=0.97,
                ),
                memory_index={},
            ),
            _analyze_step_record(
                seed=7,
                mode_name="voi_memory_v1",
                step_record=_tool_step_record(
                    action="ASK_VERIFY_1",
                    scan_eval=0.80,
                    verify_eval=0.92,
                ),
                memory_index={},
            ),
        ]
        by_profile = _group_step_rows(rows, "tool_profile")
        by_subtype = _group_step_rows(rows, "tool_subtype")
        self.assertIn("verify_necessary", by_profile)
        self.assertIn("scan_then_verify", by_subtype)

    def test_summarize_step_rows_aggregates_tool_metrics(self) -> None:
        rows = [
            _analyze_step_record(
                seed=7,
                mode_name="voi_memory_v1",
                step_record=_tool_step_record(
                    action="ASK_SCAN_1",
                    scan_eval=0.90,
                    verify_eval=0.97,
                ),
                memory_index={},
            ),
            _analyze_step_record(
                seed=7,
                mode_name="voi_memory_v1",
                step_record=_tool_step_record(
                    action="ASK_VERIFY_1",
                    scan_eval=0.96,
                    verify_eval=0.91,
                ),
                memory_index={},
            ),
            _analyze_step_record(
                seed=7,
                mode_name="voi_memory_v1",
                step_record=_tool_step_record(
                    action="ASK_VERIFY_1",
                    scan_eval=0.80,
                    verify_eval=0.92,
                ),
                memory_index={},
            ),
        ]
        summary = _summarize_step_rows(rows)
        self.assertEqual(1, summary["selected_scan_count"])
        self.assertEqual(2, summary["selected_verify_count"])
        self.assertEqual(3, summary["tool_choice_step_count"])
        self.assertAlmostEqual(0.3333, float(summary["scan_usage_rate"]), places=4)
        self.assertAlmostEqual(0.6667, float(summary["verify_usage_rate"]), places=4)
        self.assertAlmostEqual(0.3333, float(summary["tool_precision"]), places=4)
        self.assertAlmostEqual(0.6667, float(summary["wrong_tool_rate"]), places=4)
        self.assertAlmostEqual(0.0167, float(summary["over_tool_cost_mean_per_step"]), places=4)
        self.assertAlmostEqual(0.0233, float(summary["under_tool_cost_mean_per_step"]), places=4)
        self.assertIn("final_oracle_family_alignment_rate", summary)
        self.assertIn("selector_family_override_rate", summary)
        self.assertIn("selected_vs_oracle_family_mismatch_rate", summary)
        self.assertIn("surface_family_distortion_rate", summary)
        self.assertIn("best_scan_canonical_surface_rate", summary)
        self.assertIn("best_verify_canonical_surface_rate_before", summary)
        self.assertIn("best_verify_canonical_surface_rate_after", summary)
        self.assertIn("verify_candidate_exposed_rate", summary)
        self.assertIn("best_verify_missing_reason_generated_but_pruned_rate", summary)

    def test_classify_pre_selector_block_uses_ask_gate_reason(self) -> None:
        source, reason = _classify_pre_selector_block(
            ask_gate_enabled=True,
            ask_gate_allowed=False,
            ask_gate_reason="strong_negative_voi",
            pre_selector_has_direct_evidence=True,
            pre_selector_filtered_pairs_empty=False,
        )
        self.assertEqual("ask_gate", source)
        self.assertEqual("strong_negative_voi", reason)

    def test_classify_pre_selector_block_preserves_borderline_block(self) -> None:
        source, reason = _classify_pre_selector_block(
            ask_gate_enabled=True,
            ask_gate_allowed=False,
            ask_gate_reason="borderline_block",
            pre_selector_has_direct_evidence=True,
            pre_selector_filtered_pairs_empty=False,
        )
        self.assertEqual("ask_gate", source)
        self.assertEqual("borderline_block", reason)

    def test_classify_pre_selector_block_requires_strict_blocked_fallback_original(self) -> None:
        source, reason = _classify_pre_selector_block(
            ask_gate_enabled=True,
            ask_gate_allowed=False,
            ask_gate_reason="blocked_fallback_original",
            pre_selector_has_direct_evidence=True,
            pre_selector_filtered_pairs_empty=False,
        )
        self.assertEqual("ask_gate", source)
        self.assertEqual("unknown", reason)

        source, reason = _classify_pre_selector_block(
            ask_gate_enabled=True,
            ask_gate_allowed=False,
            ask_gate_reason="blocked_fallback_original",
            pre_selector_has_direct_evidence=True,
            pre_selector_filtered_pairs_empty=True,
        )
        self.assertEqual("ask_gate", source)
        self.assertEqual("blocked_fallback_original", reason)

    def test_classify_pre_selector_block_falls_back_to_unknown_without_direct_evidence(self) -> None:
        source, reason = _classify_pre_selector_block(
            ask_gate_enabled=True,
            ask_gate_allowed=False,
            ask_gate_reason="strong_negative_voi",
            pre_selector_has_direct_evidence=False,
            pre_selector_filtered_pairs_empty=False,
        )
        self.assertEqual("unknown", source)
        self.assertEqual("unknown", reason)

    def test_summarize_pre_selector_pruning_reports_verify_vs_scan_and_direct_evidence(self) -> None:
        step_rows_by_mode = {
            "voi_memory_v1": [
                _pre_selector_step_row(
                    ask_gate_reason="strong_negative_voi",
                    ask_gate_allowed=False,
                    pre_selector_has_direct_evidence=True,
                    pre_selector_filter_applied=True,
                    pre_selector_filtered_pairs_empty=False,
                    verify_entered_selector=False,
                    tool_subtype="verify_first",
                ),
                _pre_selector_step_row(
                    ask_gate_reason="strong_positive_voi",
                    ask_gate_allowed=True,
                    pre_selector_has_direct_evidence=False,
                    pre_selector_filter_applied=False,
                    pre_selector_filtered_pairs_empty=False,
                    verify_entered_selector=True,
                    tool_subtype="scan_then_verify",
                ),
            ]
        }
        rows = _collect_pre_selector_pruning_rows(step_rows_by_mode)["voi_memory_v1"]
        summary = _summarize_pre_selector_pruning(
            step_rows=step_rows_by_mode["voi_memory_v1"],
            verify_survival_summary={
                "pre_selector_filter_denominator": 2,
                "pre_selector_filter_applicable_count": 2,
                "pre_selector_filter_retained_count": 1,
                "pre_selector_filter_retained_rate": 0.5,
            },
            pre_selector_rows=rows,
        )
        self.assertEqual(1, summary["filter_rule_count"])
        self.assertEqual(0, summary["filter_rule_without_direct_gate_evidence"])
        self.assertEqual(
            {"strong_negative_voi": 1},
            summary["filter_rule_ask_gate_reason_counts"],
        )
        self.assertEqual(
            {"verify_first": 1},
            summary["filter_rule_tool_subtype_counts"],
        )
        self.assertEqual(
            2,
            summary["verify_vs_scan_pre_selector_survival"]["verify"]["pre_selector_applicable_count"],
        )
        self.assertEqual(
            1,
            summary["verify_vs_scan_pre_selector_survival"]["verify"]["pre_selector_retained_count"],
        )
        self.assertEqual(
            0,
            summary["verify_vs_scan_pre_selector_survival"]["scan"]["pre_selector_retained_count"],
        )

    def test_best_surface_eval_for_action_type_reads_logged_candidates(self) -> None:
        step_record = _blocked_audit_step_record(
            choose_action="CHOOSE_B",
            choose_eval=0.0,
            surfaced_ask_action="ASK_VERIFY_1",
            surfaced_ask_eval=0.55,
        )
        best_choose = _best_surface_eval_for_action_type(step_record, action_type="CHOOSE")
        best_ask = _best_surface_eval_for_action_type(step_record, action_type="ASK")
        self.assertEqual("CHOOSE_B", best_choose["action"])
        self.assertAlmostEqual(0.0, float(best_choose["eval_score"]), places=4)
        self.assertEqual("ASK_VERIFY_1", best_ask["action"])
        self.assertAlmostEqual(0.55, float(best_ask["eval_score"]), places=4)

    def test_classify_blocked_step_handles_missing_surfaced_ask(self) -> None:
        self.assertEqual(
            "task_pressure",
            _classify_blocked_step(
                oracle_best_ask_eval=0.95,
                best_choose_eval=1.0,
                best_surfaced_ask_eval=None,
            ),
        )
        self.assertEqual(
            "surfaced_ask_deficit",
            _classify_blocked_step(
                oracle_best_ask_eval=0.55,
                best_choose_eval=0.0,
                best_surfaced_ask_eval=None,
            ),
        )
        self.assertEqual(
            "hard_cutoff_candidate",
            _classify_blocked_step(
                oracle_best_ask_eval=0.55,
                best_choose_eval=0.0,
                best_surfaced_ask_eval=0.55,
            ),
        )

    def test_collect_blocked_step_audit_rows_classifies_surfaced_deficit(self) -> None:
        rows_by_mode = _collect_blocked_step_audit_rows(
            {
                "voi_memory_v1": [
                    {
                        "seed": 7,
                        "mode": "voi_memory_v1",
                        "step_record": _blocked_audit_step_record(
                            choose_action="CHOOSE_B",
                            choose_eval=0.0,
                        ),
                        "task_config": _oracle_gap_task(),
                        "prior_actions": [],
                    }
                ]
            },
            {"voi_memory_v1": [_blocked_step_row()]},
        )
        rows = rows_by_mode["voi_memory_v1"]
        self.assertEqual(1, len(rows))
        self.assertEqual("surfaced_ask_deficit", rows[0]["blocked_step_class"])
        self.assertTrue(rows[0]["best_surfaced_ask_missing"])
        self.assertGreater(float(rows[0]["oracle_ask_gain"]), 0.0)
        self.assertIsNone(rows[0]["surface_gap"])
        self.assertIn("Asked clues:", str(rows[0]["oracle_best_ask_observation_text"]))

    def test_summarize_blocked_step_audit_routes_task_pressure(self) -> None:
        summary = _summarize_blocked_step_audit(
            [
                {"blocked_step_class": "task_pressure", "direct_choose_correct": True, "surface_gap": None, "best_surfaced_ask_missing": False},
                {"blocked_step_class": "task_pressure", "direct_choose_correct": True, "surface_gap": 0.0, "best_surfaced_ask_missing": False},
                {"blocked_step_class": "surfaced_ask_deficit", "direct_choose_correct": False, "surface_gap": None, "best_surfaced_ask_missing": True},
            ]
        )
        self.assertEqual(3, summary["blocked_step_count"])
        self.assertAlmostEqual(0.6667, float(summary["task_pressure_rate"]), places=4)
        self.assertEqual("task_pressure", summary["route_target"])
        self.assertAlmostEqual(0.6667, float(summary["direct_choose_correct_rate"]), places=4)

    def test_render_diagnosis_report_includes_tool_metrics(self) -> None:
        tool_overall = {
            "selected_tool_count": 3,
            "scan_usage_rate": 0.3333,
            "verify_usage_rate": 0.6667,
            "tool_choice_step_count": 3,
            "tool_precision": 0.3333,
            "wrong_tool_rate": 0.6667,
            "over_tool_cost_mean_per_step": 0.0167,
            "under_tool_cost_mean_per_step": 0.0233,
            "oracle_tool_choice_step_count": 3,
            "candidate_backend_template_rate": 1.0,
            "candidate_backend_union_rate": 0.0,
            "oracle_family_gap_mean": 0.08,
            "final_oracle_family_alignment_rate": 0.6667,
            "final_oracle_scan_coverage_rate": 1.0,
            "final_oracle_verify_coverage_rate": 0.6667,
            "selected_vs_oracle_family_mismatch_rate": 0.3333,
            "choose_swallow_rate_given_oracle_tool_choice": 0.1111,
            "surface_family_distortion_rate": 0.2222,
            "policy_family_error_rate": 0.1111,
            "best_verify_canonical_surface_rate_before": 0.25,
            "best_scan_canonical_surface_rate": 0.5,
            "best_scan_fixed_surface_rate": 0.25,
            "best_scan_ask_only_surface_rate": 0.0,
            "best_scan_multi_step_surface_rate": 0.25,
            "best_verify_canonical_surface_rate": 0.75,
            "best_verify_canonical_surface_rate_after": 0.75,
            "best_verify_fixed_surface_rate": 0.25,
            "best_verify_ask_only_surface_rate": 0.0,
            "best_verify_multi_step_surface_rate": 0.0,
            "verify_candidate_exposed_rate": 0.6667,
            "best_verify_missing_reason_not_generated_rate": 0.0,
            "best_verify_missing_reason_generated_but_pruned_rate": 0.1111,
            "best_verify_missing_reason_generated_noncanonical_rate": 0.0,
            "best_verify_missing_reason_dominated_after_scoring_rate": 0.2222,
            "selector_family_override_rate": 0.0,
            "regretful_ask_rate": 0.0,
            "choose_regret_rate": 0.0,
            "selected_ask_rate": 1.0,
            "over_ask_cost_mean_per_step": 0.0,
            "under_ask_cost_mean_per_step": 0.0,
            "selected_choose_without_prior_ask_rate": 0.0,
            "ask_gate_step_count": 3,
            "ask_gate_block_rate": 0.0,
            "ask_gate_allow_rate_given_positive_ask_advantage": 1.0,
            "ask_gate_block_rate_given_nonpositive_ask_advantage": 0.0,
            "retrieved_positive_evidence_rate": 0.0,
            "retrieved_caution_evidence_rate": 0.0,
        }
        diagnosis = {
            "config": {"run_root": "tmp/tool_run", "seeds": [7]},
            "modes": {
                "full": {
                    "overall": dict(tool_overall),
                    "episodes": {"success_rate": 1.0, "avg_reward": 0.95, "avg_steps": 1.5, "avg_asks": 1.0},
                },
                "no_reflection": {
                    "overall": dict(tool_overall),
                    "episodes": {"success_rate": 1.0, "avg_reward": 0.94, "avg_steps": 1.6, "avg_asks": 1.0},
                },
                "voi_memory_v1": {
                    "overall": dict(tool_overall),
                    "episodes": {"success_rate": 1.0, "avg_reward": 0.99, "avg_steps": 1.2, "avg_asks": 0.5},
                },
            },
            "verify_survival": {
                "full": {
                    "verify_lineage_count": 4,
                    "verify_generated_rate": 0.75,
                    "verify_normalized_rate": 0.75,
                    "verify_retained_after_family_merge_rate": 0.75,
                    "verify_retained_after_candidate_cap_rate": 0.75,
                    "verify_retained_after_dedup_rate": 0.50,
                    "verify_retained_before_selector_rate": 0.50,
                    "verify_entered_selector_rate": 0.25,
                    "verify_injected_before_selector_rate": 0.00,
                    "verify_shape_replacement_rate": 0.50,
                    "generated_denominator": 4,
                    "generated_applicable_count": 4,
                    "generated_retained_count": 3,
                    "generated_retained_rate": 0.75,
                    "family_merge_denominator": 4,
                    "family_merge_applicable_count": 3,
                    "family_merge_retained_count": 3,
                    "family_merge_retained_rate": 1.0,
                    "normalization_merge_denominator": 4,
                    "normalization_merge_applicable_count": 3,
                    "normalization_merge_retained_count": 3,
                    "normalization_merge_retained_rate": 1.0,
                    "dedup_denominator": 4,
                    "dedup_applicable_count": 3,
                    "dedup_retained_count": 2,
                    "dedup_retained_rate": 0.6667,
                    "candidate_cap_denominator": 4,
                    "candidate_cap_applicable_count": 4,
                    "candidate_cap_retained_count": 3,
                    "candidate_cap_retained_rate": 0.75,
                    "pre_selector_filter_denominator": 4,
                    "pre_selector_filter_applicable_count": 2,
                    "pre_selector_filter_retained_count": 1,
                    "pre_selector_filter_retained_rate": 0.5,
                    "selector_entry_denominator": 4,
                    "selector_entry_applicable_count": 2,
                    "selector_entry_retained_count": 1,
                    "selector_entry_retained_rate": 0.5,
                    "verify_pruned_stage_generated_rate": 0.25,
                    "verify_pruned_stage_family_merge_rate": 0.0,
                    "verify_pruned_stage_normalization_merge_rate": 0.0,
                    "verify_pruned_stage_dedup_rate": 0.25,
                    "verify_pruned_stage_candidate_cap_rate": 0.25,
                    "verify_pruned_stage_pre_selector_filter_rate": 0.25,
                    "verify_pruned_stage_unknown_rate": 0.0,
                    "verify_pruned_reason_shape_replaced_by_equivalent_rate": 0.0,
                    "verify_pruned_reason_dominated_within_family_rate": 0.0,
                    "verify_pruned_reason_dominated_cross_family_rate": 0.25,
                    "verify_pruned_reason_deduplicated_rate": 0.25,
                    "verify_pruned_reason_budget_cap_rate": 0.0,
                    "verify_pruned_reason_filter_rule_rate": 0.25,
                    "verify_pruned_reason_unknown_rate": 0.25,
                },
                "selector_penalty": {
                    "verify_lineage_count": 4,
                    "verify_generated_rate": 0.75,
                    "verify_normalized_rate": 0.75,
                    "verify_retained_after_family_merge_rate": 0.75,
                    "verify_retained_after_candidate_cap_rate": 0.75,
                    "verify_retained_after_dedup_rate": 0.50,
                    "verify_retained_before_selector_rate": 0.50,
                    "verify_entered_selector_rate": 0.25,
                    "verify_injected_before_selector_rate": 0.00,
                    "verify_shape_replacement_rate": 0.50,
                    "generated_denominator": 4,
                    "generated_applicable_count": 4,
                    "generated_retained_count": 3,
                    "generated_retained_rate": 0.75,
                    "family_merge_denominator": 4,
                    "family_merge_applicable_count": 3,
                    "family_merge_retained_count": 3,
                    "family_merge_retained_rate": 1.0,
                    "normalization_merge_denominator": 4,
                    "normalization_merge_applicable_count": 3,
                    "normalization_merge_retained_count": 3,
                    "normalization_merge_retained_rate": 1.0,
                    "dedup_denominator": 4,
                    "dedup_applicable_count": 3,
                    "dedup_retained_count": 2,
                    "dedup_retained_rate": 0.6667,
                    "candidate_cap_denominator": 4,
                    "candidate_cap_applicable_count": 4,
                    "candidate_cap_retained_count": 3,
                    "candidate_cap_retained_rate": 0.75,
                    "pre_selector_filter_denominator": 4,
                    "pre_selector_filter_applicable_count": 2,
                    "pre_selector_filter_retained_count": 1,
                    "pre_selector_filter_retained_rate": 0.5,
                    "selector_entry_denominator": 4,
                    "selector_entry_applicable_count": 2,
                    "selector_entry_retained_count": 1,
                    "selector_entry_retained_rate": 0.5,
                    "verify_pruned_stage_generated_rate": 0.25,
                    "verify_pruned_stage_family_merge_rate": 0.0,
                    "verify_pruned_stage_normalization_merge_rate": 0.0,
                    "verify_pruned_stage_dedup_rate": 0.25,
                    "verify_pruned_stage_candidate_cap_rate": 0.25,
                    "verify_pruned_stage_pre_selector_filter_rate": 0.25,
                    "verify_pruned_stage_unknown_rate": 0.0,
                    "verify_pruned_reason_shape_replaced_by_equivalent_rate": 0.0,
                    "verify_pruned_reason_dominated_within_family_rate": 0.0,
                    "verify_pruned_reason_dominated_cross_family_rate": 0.25,
                    "verify_pruned_reason_deduplicated_rate": 0.25,
                    "verify_pruned_reason_budget_cap_rate": 0.0,
                    "verify_pruned_reason_filter_rule_rate": 0.25,
                    "verify_pruned_reason_unknown_rate": 0.25,
                },
                "voi_memory_v1": {
                    "verify_lineage_count": 4,
                    "verify_generated_rate": 0.50,
                    "verify_normalized_rate": 0.50,
                    "verify_retained_after_family_merge_rate": 0.50,
                    "verify_retained_after_candidate_cap_rate": 0.50,
                    "verify_retained_after_dedup_rate": 0.50,
                    "verify_retained_before_selector_rate": 0.50,
                    "verify_entered_selector_rate": 0.25,
                    "verify_injected_before_selector_rate": 0.0,
                    "verify_shape_replacement_rate": 0.25,
                    "generated_denominator": 4,
                    "generated_applicable_count": 4,
                    "generated_retained_count": 2,
                    "generated_retained_rate": 0.5,
                    "family_merge_denominator": 4,
                    "family_merge_applicable_count": 2,
                    "family_merge_retained_count": 2,
                    "family_merge_retained_rate": 1.0,
                    "normalization_merge_denominator": 4,
                    "normalization_merge_applicable_count": 2,
                    "normalization_merge_retained_count": 2,
                    "normalization_merge_retained_rate": 1.0,
                    "dedup_denominator": 4,
                    "dedup_applicable_count": 2,
                    "dedup_retained_count": 2,
                    "dedup_retained_rate": 1.0,
                    "candidate_cap_denominator": 4,
                    "candidate_cap_applicable_count": 4,
                    "candidate_cap_retained_count": 2,
                    "candidate_cap_retained_rate": 0.5,
                    "pre_selector_filter_denominator": 4,
                    "pre_selector_filter_applicable_count": 2,
                    "pre_selector_filter_retained_count": 1,
                    "pre_selector_filter_retained_rate": 0.5,
                    "selector_entry_denominator": 4,
                    "selector_entry_applicable_count": 2,
                    "selector_entry_retained_count": 1,
                    "selector_entry_retained_rate": 0.5,
                    "verify_pruned_stage_generated_rate": 0.25,
                    "verify_pruned_stage_family_merge_rate": 0.0,
                    "verify_pruned_stage_normalization_merge_rate": 0.0,
                    "verify_pruned_stage_dedup_rate": 0.25,
                    "verify_pruned_stage_candidate_cap_rate": 0.25,
                    "verify_pruned_stage_pre_selector_filter_rate": 0.25,
                    "verify_pruned_stage_unknown_rate": 0.0,
                    "verify_pruned_reason_shape_replaced_by_equivalent_rate": 0.0,
                    "verify_pruned_reason_dominated_within_family_rate": 0.0,
                    "verify_pruned_reason_dominated_cross_family_rate": 0.25,
                    "verify_pruned_reason_deduplicated_rate": 0.25,
                    "verify_pruned_reason_budget_cap_rate": 0.0,
                    "verify_pruned_reason_filter_rule_rate": 0.25,
                    "verify_pruned_reason_unknown_rate": 0.25,
                },
            },
            "pre_selector_pruning": {
                "voi_memory_v1": {
                    "claim_boundary": "attribute block reason only",
                    "primary_evidence_fields": [
                        "verify_pruned_stage",
                        "verify_pruned_reason",
                        "verify_retained_before_selector",
                        "verify_entered_selector",
                        "ask_gate_enabled",
                        "ask_gate_allowed",
                        "ask_gate_reason",
                        "pre_selector_block_source",
                        "pre_selector_block_reason",
                    ],
                    "secondary_context_fields": [
                        "ask_gate_score",
                        "ask_advantage",
                        "best_ask_family",
                        "best_ask_eval",
                        "best_choose_eval",
                        "voi_ratio",
                        "uncertainty",
                        "base_uncertainty",
                        "reliability_uncertainty",
                        "evidence_conflict",
                        "memory_signal",
                        "helpful_ask",
                        "choose_regret",
                        "regretful_ask",
                        "choose_helpful",
                    ],
                    "pre_selector_denominator": 4,
                    "pre_selector_applicable_count": 2,
                    "pre_selector_retained_count": 1,
                    "pre_selector_retained_rate": 0.5,
                    "filter_rule_count": 1,
                    "filter_rule_without_direct_gate_evidence": 0,
                    "filter_rule_ask_gate_reason_counts": {"strong_negative_voi": 1},
                    "filter_rule_tool_subtype_counts": {"verify_first": 1},
                    "secondary_context_comparison": {
                        "ask_gate_score": {"filter_rule_mean": -0.2, "entered_selector_mean": 0.4},
                    },
                    "verify_vs_scan_pre_selector_survival": {
                        "verify": {
                            "pre_selector_applicable_count": 2,
                            "pre_selector_retained_count": 1,
                            "pre_selector_retained_rate": 0.5,
                        },
                        "scan": {
                            "pre_selector_applicable_count": 2,
                            "pre_selector_retained_count": 0,
                            "pre_selector_retained_rate": 0.0,
                        },
                    },
                }
            },
            "blocked_step_audit": {
                "voi_memory_v1": {
                    "claim_boundary": "blocked-step classification only",
                    "blocked_step_count": 5,
                    "task_pressure_count": 3,
                    "task_pressure_rate": 0.6,
                    "surfaced_ask_deficit_count": 1,
                    "surfaced_ask_deficit_rate": 0.2,
                    "hard_cutoff_candidate_count": 1,
                    "hard_cutoff_candidate_rate": 0.2,
                    "direct_choose_correct_rate": 0.6,
                    "surface_gap_count": 2,
                    "surface_gap_mean": 0.15,
                    "surface_gap_median": 0.15,
                    "surface_gap_positive_rate": 1.0,
                    "route_target": "task_pressure",
                    "route_threshold": 0.6,
                    "route_requires_new_experiment": False,
                }
            },
            "subgroups": {"difficulty": {}, "noise_level": {}, "step_index": {}, "tool_profile": {}, "tool_subtype": {}},
            "reflection_bias": {
                "full": {
                    "avoid_actions": {"CHOOSE": {"rate": 0.0, "count": 0}, "ASK": {"rate": 0.0, "count": 0}, "EMPTY": {"rate": 1.0, "count": 1}, "MIXED": {"rate": 0.0, "count": 0}},
                    "z_delta": {
                        "info_seeking": {"mean": 0.0, "abs_mean": 0.0, "positive_rate": 0.0, "negative_rate": 0.0},
                        "cost_sensitivity": {"mean": 0.0, "abs_mean": 0.0, "positive_rate": 0.0, "negative_rate": 0.0},
                        "confidence_threshold": {"mean": 0.0, "abs_mean": 0.0, "positive_rate": 0.0, "negative_rate": 0.0},
                    },
                    "subsequent_correlations": {
                        "avoid_choose_flag": {"next_ask_count": 0.0, "next_reward": 0.0},
                        "z_delta.confidence_threshold": {"next_ask_count": 0.0},
                        "z_delta.info_seeking": {"next_ask_count": 0.0},
                    },
                    "memory_alignment": {
                        "selected_ask_with_choose_suppressing_memory_rate": 0.0,
                        "regretful_ask_with_choose_suppressing_memory_rate": 0.0,
                    },
                }
            },
            "recommendation": {"recommended_intervention": "voi_memory_v1", "reasons": ["best overall"]},
            "routing": {
                "routing_modes": ["full", "selector_penalty", "voi_memory_v1"],
                "most_likely_layer": "candidate_coverage",
                "candidate_related_share": 0.75,
                "policy_family_error_share": 0.10,
                "key_symptoms": ["candidate-related mismatch dominates"],
                "key_metrics": [{"name": "candidate_related_share", "value": 0.75}],
                "alternative_explanations": ["runtime_backend"],
            },
        }
        report = _render_diagnosis_report(diagnosis)
        self.assertIn("## Tool-Use Summary", report)
        self.assertIn("## Runtime Tool Diagnosis", report)
        self.assertIn("## Candidate-Layer Diagnosis", report)
        self.assertIn("## Verify Lineage Survival Audit", report)
        self.assertIn("## Pre-Selector Pruning Attribution", report)
        self.assertIn("- Full scan usage rate: 0.333", report)
        self.assertIn("- Tool precision: 0.333 (tool_choice_step_count=3)", report)
        self.assertIn("- Wrong-tool rate: 0.667", report)
        self.assertIn("- Oracle tool-choice step count (ask-opportunity proxy): 3.000", report)
        self.assertIn("- Best-verify canonical surface before: 0.250", report)
        self.assertIn("- Verify candidate exposed rate: 0.667", report)
        self.assertIn("- Best-verify missing reasons:", report)
        self.assertIn("- Generated: denominator=4, applicable=4, retained=3, retained_rate=0.750", report)
        self.assertIn("- Pruned stage distribution:", report)
        self.assertIn("- Primary evidence fields:", report)
        self.assertIn("- `voi_memory_v1` pre-selector summary: denominator=4, applicable_count=2, retained_count=1, retained_rate=0.500", report)
        self.assertIn("- Verify vs scan pre-selector survival:", report)
        self.assertIn("filter_rule_without_direct_gate_evidence=0", report)
        self.assertIn("## Blocked ASK Value Audit", report)
        self.assertIn("- Blocked step count: 5", report)
        self.assertIn("- Suggested next route: task_pressure", report)
        audit = _render_pre_selector_pruning_audit(diagnosis)
        self.assertIn("# Pre-Selector Pruning Audit", audit)
        self.assertIn("- Primary evidence fields:", audit)
        self.assertIn("- Secondary context fields:", audit)
        self.assertIn("- Verify vs scan pre-selector survival:", audit)
        blocked_audit = _render_blocked_step_audit(diagnosis)
        self.assertIn("# Blocked ASK Value Audit", blocked_audit)
        self.assertIn("- task_pressure: count=3, rate=0.600", blocked_audit)
        self.assertIn("- Suggested next route: task_pressure", blocked_audit)


if __name__ == "__main__":
    unittest.main()
