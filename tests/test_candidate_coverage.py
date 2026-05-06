from __future__ import annotations

import unittest

from agent.candidate_coverage import (
    build_candidate_layer_bundle,
    build_verify_lineage_survival,
    candidate_surface_shape,
    infer_best_verify_missing_reason,
    infer_verify_candidate_exposed,
)


class CandidateCoverageTests(unittest.TestCase):
    def test_build_candidate_layer_bundle_normalizes_and_exposes_verify(self) -> None:
        bundle = build_candidate_layer_bundle(
            base_candidates=[
                {"name": "choose_a", "actions": ["CHOOSE_A"], "category": "choose_now"},
                {"name": "scan_fixed", "actions": ["ASK_SCAN_2", "CHOOSE_A"], "category": "ask_then_choose_fixed"},
            ],
            available_actions=["ASK_SCAN_2", "ASK_VERIFY_3"],
            template_candidates=[
                {"name": "template_verify", "actions": ["ASK_VERIFY_3", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
        )
        after_candidates = bundle["candidates_after_normalization"]
        action_sets = {tuple(candidate["actions"]) for candidate in after_candidates}
        self.assertIn(("ASK_SCAN_2", "CHOOSE_BEST"), action_sets)
        self.assertIn(("ASK_VERIFY_3", "CHOOSE_BEST"), action_sets)
        self.assertEqual(0, bundle["candidate_count_by_family_before_normalization"]["verify"])
        self.assertEqual(1, bundle["candidate_count_by_family_after_normalization"]["verify"])

    def test_verify_missing_reason_reports_generated_but_pruned(self) -> None:
        reason = infer_best_verify_missing_reason(
            remaining_verify_actions=["ASK_VERIFY_1"],
            candidates_before_normalization=[
                {"name": "verify_only", "actions": ["ASK_VERIFY_1"], "category": "ask_only"},
            ],
            candidates_after_normalization=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            selector_candidates=[
                {"name": "choose_a", "actions": ["CHOOSE_A"], "category": "choose_now"},
            ],
            selector_evaluations=[
                {"candidate_name": "choose_a", "score": 0.8},
            ],
        )
        self.assertEqual("generated_but_pruned", reason)

    def test_verify_missing_reason_reports_dominated_after_scoring(self) -> None:
        selector_candidates = [
            {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            {"name": "scan_best", "actions": ["ASK_SCAN_2", "CHOOSE_BEST"], "category": "ask_then_choose"},
        ]
        reason = infer_best_verify_missing_reason(
            remaining_verify_actions=["ASK_VERIFY_1"],
            candidates_before_normalization=selector_candidates,
            candidates_after_normalization=selector_candidates,
            selector_candidates=selector_candidates,
            selector_evaluations=[
                {"candidate_name": "verify_best", "score": 0.4},
                {"candidate_name": "scan_best", "score": 0.9},
            ],
        )
        self.assertEqual("dominated_after_scoring", reason)
        self.assertTrue(infer_verify_candidate_exposed(selector_candidates))
        self.assertEqual(
            "canonical_choose_best",
            candidate_surface_shape(selector_candidates[0]),
        )

    def test_verify_lineage_survival_groups_variants_by_first_action(self) -> None:
        rows = build_verify_lineage_survival(
            remaining_verify_actions=["ASK_VERIFY_1"],
            candidate_backend="openai_responses_plus_template",
            candidate_limit=5,
            template_candidates=[
                {"name": "verify_fixed", "actions": ["ASK_VERIFY_1", "CHOOSE_A"], "category": "ask_then_choose_fixed"},
            ],
            llm_candidates=[
                {"name": "verify_only", "actions": ["ASK_VERIFY_1"], "category": "ask_only"},
            ],
            source_candidates=[
                {"name": "verify_fixed", "actions": ["ASK_VERIFY_1", "CHOOSE_A"], "category": "ask_then_choose_fixed"},
                {"name": "verify_only", "actions": ["ASK_VERIFY_1"], "category": "ask_only"},
            ],
            normalized_candidates_pre_dedup=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
                {"name": "verify_best_dup", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            deduped_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            final_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            selector_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            ask_gate_allowed=True,
        )
        self.assertEqual(1, len(rows))
        self.assertTrue(rows[0]["verify_generated"])
        self.assertTrue(rows[0]["verify_entered_selector"])
        self.assertTrue(rows[0]["verify_shape_replacement_detected"])
        self.assertEqual("none", rows[0]["verify_pruned_stage"])

    def test_verify_lineage_survival_marks_source_side_budget_cap(self) -> None:
        rows = build_verify_lineage_survival(
            remaining_verify_actions=["ASK_VERIFY_1"],
            candidate_backend="openai_responses",
            candidate_limit=1,
            llm_candidates=[
                {"name": "choose_a", "actions": ["CHOOSE_A"], "category": "choose_now"},
            ],
            source_candidates=[
                {"name": "choose_a", "actions": ["CHOOSE_A"], "category": "choose_now"},
            ],
            final_candidates=[
                {"name": "choose_a", "actions": ["CHOOSE_A"], "category": "choose_now"},
            ],
            selector_candidates=[
                {"name": "choose_a", "actions": ["CHOOSE_A"], "category": "choose_now"},
            ],
        )
        self.assertEqual("candidate_cap", rows[0]["verify_pruned_stage"])
        self.assertEqual("dominated_cross_family", rows[0]["verify_pruned_reason"])

    def test_verify_lineage_survival_marks_dedup_loss(self) -> None:
        rows = build_verify_lineage_survival(
            remaining_verify_actions=["ASK_VERIFY_1"],
            candidate_backend="openai_responses",
            candidate_limit=5,
            llm_candidates=[
                {"name": "verify_fixed", "actions": ["ASK_VERIFY_1", "CHOOSE_A"], "category": "ask_then_choose_fixed"},
            ],
            source_candidates=[
                {"name": "verify_fixed", "actions": ["ASK_VERIFY_1", "CHOOSE_A"], "category": "ask_then_choose_fixed"},
            ],
            normalized_candidates_pre_dedup=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            deduped_candidates=[],
            final_candidates=[],
            selector_candidates=[],
        )
        self.assertEqual("dedup", rows[0]["verify_pruned_stage"])
        self.assertEqual("deduplicated", rows[0]["verify_pruned_reason"])

    def test_verify_lineage_survival_marks_pre_selector_filter(self) -> None:
        rows = build_verify_lineage_survival(
            remaining_verify_actions=["ASK_VERIFY_1"],
            candidate_backend="openai_responses",
            candidate_limit=5,
            llm_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            source_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            normalized_candidates_pre_dedup=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            deduped_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            final_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            selector_candidates=[],
            ask_gate_allowed=False,
            pre_selector_filter_evidence=True,
        )
        self.assertEqual("pre_selector_filter", rows[0]["verify_pruned_stage"])
        self.assertEqual("filter_rule", rows[0]["verify_pruned_reason"])

    def test_verify_lineage_survival_marks_unknown_without_direct_pre_selector_evidence(self) -> None:
        rows = build_verify_lineage_survival(
            remaining_verify_actions=["ASK_VERIFY_1"],
            candidate_backend="openai_responses",
            candidate_limit=5,
            llm_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            source_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            normalized_candidates_pre_dedup=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            deduped_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            final_candidates=[
                {"name": "verify_best", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
            ],
            selector_candidates=[],
            ask_gate_allowed=False,
            pre_selector_filter_evidence=False,
        )
        self.assertEqual("pre_selector_filter", rows[0]["verify_pruned_stage"])
        self.assertEqual("unknown", rows[0]["verify_pruned_reason"])


if __name__ == "__main__":
    unittest.main()
