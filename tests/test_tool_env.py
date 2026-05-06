from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from unittest import mock

from agent.counterfactual import belief_from_observation, generate_candidates
from agent.llm import OpenAIResponsesClient
from agent.self_model import SelfModel
from data.generate_tasks import (
    FAMILY_CONFLICT_DELTA,
    REFERENCE_REPAIR_LANE,
    STEP_LOCAL_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE,
    STEP_LOCAL_PREFLIGHT_POSITIVE_ASK_GAIN_MIN_RATE,
    TASK_PRESSURE_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE,
    V3_DECEPTIVE_SCAN_SHARE_MIN,
    V3_FAMILY_CONFLICT_DELTA,
    V3_SCAN_ONLY_UNSAFE_RATE_MIN,
    V4_DECEPTIVE_SCAN_SHARE_MIN,
    V4_FAMILY_CONFLICT_DELTA,
    V4_FAMILY_CONFLICT_POSITIVE_RATE_MIN,
    V4_SCAN_ONLY_UNSAFE_RATE_MIN,
    _verify_first_tool_clues,
    generate_tasks_file,
    generate_tasks,
    summarize_toolized_profile,
    validate_toolized_task,
)
from env.toytextmdp import ToyTextMDP


class _PromptCaptureClient(OpenAIResponsesClient):
    def __init__(self) -> None:
        super().__init__()
        self.api_key = "test-key"
        self.captured_input: str | None = None

    def _request_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, object],
        max_output_tokens: int,
    ) -> dict[str, object] | None:
        del instructions, schema_name, schema, max_output_tokens
        self.captured_input = input_text
        return {
            "candidates": [
                {"name": "scan", "actions": ["ASK_SCAN_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
                {"name": "verify", "actions": ["ASK_VERIFY_1", "CHOOSE_BEST"], "category": "ask_then_choose"},
                {"name": "choose", "actions": ["CHOOSE_A"], "category": "choose_now"},
            ]
        }


class ToolizedTaskGenerationTests(unittest.TestCase):
    def test_toolized_tasks_emit_scan_and_verify_fields(self) -> None:
        task = generate_tasks(count=1, seed=7, toolized=True)[0]
        self.assertTrue(task["toolized"])
        clue = task["clues"][0]
        self.assertIn("scan_scores", clue)
        self.assertIn("verify_scores", clue)
        self.assertIn("scan_cost", clue)
        self.assertIn("verify_cost", clue)
        self.assertIn("scan_text", clue)
        self.assertIn("verify_text", clue)

    def test_verify_necessary_profile_emits_metadata_and_summary(self) -> None:
        tasks = generate_tasks(count=10, seed=7, toolized=True, toolized_profile="verify_necessary")
        summary = summarize_toolized_profile(tasks)
        self.assertEqual(10, summary["task_count"])
        self.assertEqual(10, summary["pass_count"])
        self.assertEqual({"verify_necessary": 10}, summary["tool_profile_counts"])
        self.assertEqual({"verify_first": 4, "scan_then_verify": 6}, summary["tool_subtype_counts"])
        self.assertGreater(summary["verify_better"], 0)
        self.assertGreater(summary["scan_better"], 0)
        self.assertGreater(summary["initial_positive_canonical_ask_gain_count"], 0)
        self.assertTrue(summary["task_pressure_preflight_passes"])
        self.assertTrue(summary["step_local_preflight_passes"])
        self.assertTrue(summary["combined_preflight_passes"])
        self.assertLess(
            summary["initial_direct_choose_optimal_rate"],
            TASK_PRESSURE_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE,
        )
        self.assertGreaterEqual(
            summary["bs_positive_ask_gain_rate"],
            STEP_LOCAL_PREFLIGHT_POSITIVE_ASK_GAIN_MIN_RATE,
        )
        self.assertLessEqual(
            summary["bs_direct_choose_optimal_rate"],
            STEP_LOCAL_PREFLIGHT_DIRECT_CHOOSE_MAX_RATE,
        )
        self.assertIsNotNone(summary["ask_gain_margin_p25"])
        self.assertGreater(float(summary["ask_gain_margin_p25"]), 0.0)
        self.assertTrue(all(task["tool_profile"] == "verify_necessary" for task in tasks))
        self.assertTrue(all(task["tool_subtype"] in {"verify_first", "scan_then_verify"} for task in tasks))

    def test_family_conflict_v2_profile_emits_stress_metrics_and_deceptive_scan(self) -> None:
        tasks = generate_tasks(
            count=12,
            seed=7,
            toolized=True,
            toolized_profile="verify_necessary_family_conflict_v2",
        )
        summary = summarize_toolized_profile(tasks)
        self.assertEqual(12, summary["task_count"])
        self.assertEqual(
            {"verify_necessary_family_conflict_v2": 12},
            summary["tool_profile_counts"],
        )
        self.assertIn("deceptive_scan", summary["tool_subtype_counts"])
        self.assertEqual(round(FAMILY_CONFLICT_DELTA, 4), summary["family_conflict_delta"])
        self.assertIn("family_conflict_positive_rate", summary)
        self.assertIn("wrong_tool_opportunity_proxy_rate", summary)
        self.assertIn("scan_only_unsafe_rate", summary)
        self.assertIn("verify_rescue_rate_after_scan", summary)
        self.assertGreater(summary["family_conflict_proxy_state_count"], 0)
        self.assertGreater(summary["scan_family_proxy_state_count"], 0)
        self.assertGreater(summary["family_conflict_positive_rate"], 0.0)
        self.assertGreater(summary["wrong_tool_opportunity_proxy_rate"], 0.0)
        self.assertGreater(summary["scan_only_unsafe_rate"], 0.0)
        self.assertGreater(summary["verify_rescue_rate_after_scan"], 0.0)
        self.assertTrue(all(task["tool_profile"] == "verify_necessary_family_conflict_v2" for task in tasks))
        self.assertTrue(
            all(task["tool_subtype"] in {"verify_first", "scan_then_verify", "deceptive_scan"} for task in tasks)
        )

    def test_aggressive_stress_v3_profile_emits_v3_gate_fields(self) -> None:
        tasks = generate_tasks(
            count=20,
            seed=7,
            toolized=True,
            toolized_profile="verify_necessary_aggressive_stress_v3",
        )
        summary = summarize_toolized_profile(tasks)
        self.assertEqual(
            {"verify_necessary_aggressive_stress_v3": 20},
            summary["tool_profile_counts"],
        )
        self.assertEqual(round(V3_FAMILY_CONFLICT_DELTA, 4), summary["family_conflict_delta_used"])
        self.assertGreater(summary["family_conflict_positive_rate"], 0.0)
        self.assertGreaterEqual(summary["deceptive_scan_share"], V3_DECEPTIVE_SCAN_SHARE_MIN)
        self.assertGreaterEqual(summary["scan_only_unsafe_rate"], V3_SCAN_ONLY_UNSAFE_RATE_MIN)
        self.assertTrue(summary["v3_preflight_passes"])
        self.assertTrue(summary["combined_preflight_passes"])
        self.assertNotIn("scan_only_unsafe_rate", summary["report_only_stress_metrics"])

    def test_aggressive_stress_v4_profile_enforces_review_hard_gates(self) -> None:
        tasks = generate_tasks(
            count=20,
            seed=7,
            toolized=True,
            toolized_profile="verify_necessary_aggressive_stress_v4",
        )
        summary = summarize_toolized_profile(tasks)
        self.assertEqual(
            {"verify_necessary_aggressive_stress_v4": 20},
            summary["tool_profile_counts"],
        )
        self.assertEqual({"deceptive_scan": 12, "scan_then_verify": 4, "verify_first": 4}, summary["tool_subtype_counts"])
        self.assertEqual(round(V4_FAMILY_CONFLICT_DELTA, 4), summary["family_conflict_delta_used"])
        self.assertGreater(
            summary["family_conflict_positive_rate"],
            V4_FAMILY_CONFLICT_POSITIVE_RATE_MIN,
        )
        self.assertGreaterEqual(summary["deceptive_scan_share"], V4_DECEPTIVE_SCAN_SHARE_MIN)
        self.assertGreaterEqual(summary["scan_only_unsafe_rate"], V4_SCAN_ONLY_UNSAFE_RATE_MIN)
        self.assertGreater(summary["wrong_tool_opportunity_proxy_rate"], 0.0)
        self.assertGreater(summary["verify_rescue_rate_after_scan"], 0.0)
        self.assertTrue(summary["v4_preflight_passes"])
        self.assertTrue(summary["combined_preflight_passes"])
        self.assertIn("scan_only_unsafe_rate must be >= 0.50", summary["v4_preflight_notes"])
        self.assertTrue(all(task["tool_profile"] == "verify_necessary_aggressive_stress_v4" for task in tasks))

    def test_summary_rejects_batch_when_step_local_preflight_fails(self) -> None:
        tasks = [{"task_id": "task-0001", "toolized": True, "tool_profile": "verify_necessary", "tool_subtype": "verify_first"}]
        weak_validation = {
            "task_id": "task-0001",
            "tool_profile": "verify_necessary",
            "tool_subtype": "verify_first",
            "passes": True,
            "failures": [],
            "notes": [],
            "initial": {
                "best_family": "verify",
                "best_scan_gain_vs_template_choose": 0.1,
                "best_verify_gain_vs_template_choose": 0.2,
                "best_canonical_ask_gain_vs_template_choose": 0.2,
                "template_choose_optimal": False,
            },
            "after_scan": {"qualifying_scan_actions": [], "snapshots": {}},
            "step_local": {
                "proxy_states": [
                    {"oracle_ask_gain": 0.25, "direct_choose_optimal": False},
                    {"oracle_ask_gain": -0.02, "direct_choose_optimal": True},
                    {"oracle_ask_gain": 0.01, "direct_choose_optimal": False},
                ]
            },
        }
        with mock.patch("data.generate_tasks.validate_toolized_task", return_value=weak_validation):
            summary = summarize_toolized_profile(tasks)
        self.assertTrue(summary["task_pressure_preflight_passes"])
        self.assertFalse(summary["step_local_preflight_passes"])
        self.assertFalse(summary["combined_preflight_passes"])
        self.assertLess(summary["bs_positive_ask_gain_rate"], STEP_LOCAL_PREFLIGHT_POSITIVE_ASK_GAIN_MIN_RATE)

    def test_generate_tasks_file_rejects_batch_when_step_local_preflight_fails(self) -> None:
        weak_summary = {
            "task_pressure_preflight_passes": True,
            "step_local_preflight_passes": False,
            "combined_preflight_passes": False,
        }
        with mock.patch("data.generate_tasks.summarize_toolized_profile", return_value=weak_summary):
            with self.assertRaises(RuntimeError):
                generate_tasks_file(
                    path=Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime/step_local_preflight_fail/tasks.jsonl"),
                    count=1,
                    seed=7,
                    toolized=True,
                    toolized_profile="verify_necessary",
                )

    def test_generate_tasks_file_can_freeze_reference_even_when_preflight_fails(self) -> None:
        weak_summary = {
            "task_pressure_preflight_passes": False,
            "step_local_preflight_passes": False,
            "combined_preflight_passes": False,
        }
        output_path = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime/reference_freeze/tasks.jsonl")
        with mock.patch("data.generate_tasks.summarize_toolized_profile", return_value=weak_summary):
            summary = generate_tasks_file(
                path=output_path,
                count=1,
                seed=7,
                toolized=True,
                toolized_profile="verify_necessary",
                allow_preflight_failure=True,
            )
        self.assertEqual(weak_summary, summary)

    def test_generate_tasks_file_writes_preflight_summary_to_explicit_path(self) -> None:
        output_root = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime/preflight_summary_out")
        output_path = output_root / "tasks.jsonl"
        summary_path = output_root / "task_pressure_preflight_summary.json"
        summary = generate_tasks_file(
            path=output_path,
            count=12,
            seed=7,
            toolized=True,
            toolized_profile="verify_necessary_family_conflict_v2",
            preflight_summary_out=summary_path,
        )
        self.assertIsNotNone(summary)
        self.assertTrue(summary_path.exists())
        written = summary_path.read_text(encoding="utf-8")
        self.assertIn("\"combined_preflight_passes\"", written)

    def test_reference_repair_disallows_allow_preflight_failure(self) -> None:
        with self.assertRaises(ValueError):
            generate_tasks_file(
                path=Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime/reference_repair_disallow/tasks.jsonl"),
                count=20,
                seed=11,
                toolized=True,
                toolized_profile="verify_necessary",
                allow_preflight_failure=True,
                reference_repair=True,
                repair_pool_count=40,
                repair_selection_policy="minimal_preflight_margin_fix",
            )

    def test_reference_repair_summary_fields_and_profile(self) -> None:
        output_root = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime/reference_repair_fields")
        output_path = output_root / "tasks.jsonl"
        summary_path = output_root / "task_pressure_preflight_summary.json"
        source_failed_attempt = Path(
            "C:/Users/yaobc/Documents/Playground/logs/seed11_v3_reference_confirmation_20260430/reference_pack"
        )
        base_summary = {
            "task_pressure_preflight_passes": True,
            "step_local_preflight_passes": True,
            "combined_preflight_passes": True,
            "tool_profile_counts": {"verify_necessary": 20},
            "tool_subtype_counts": {"verify_first": 8, "scan_then_verify": 12},
        }
        with mock.patch("data.generate_tasks.summarize_toolized_profile", return_value=base_summary):
            summary = generate_tasks_file(
                path=output_path,
                count=20,
                seed=11,
                toolized=True,
                toolized_profile="verify_necessary",
                reference_repair=True,
                repair_pool_count=40,
                repair_selection_policy="minimal_preflight_margin_fix",
                source_failed_attempt=source_failed_attempt,
                repair_attempt_index=1,
                preflight_summary_out=summary_path,
            )
        self.assertIsNotNone(summary)
        self.assertTrue(summary["repair_attempt"])
        self.assertEqual(REFERENCE_REPAIR_LANE, summary["repair_lane"])
        self.assertEqual(str(source_failed_attempt), summary["source_failed_attempt"])
        self.assertEqual("minimal_preflight_margin_fix", summary["selection_policy"])
        self.assertEqual(40, summary["candidate_pool_count"])
        self.assertEqual(20, summary["selected_task_count"])
        self.assertEqual(20, summary["discarded_task_count"])
        self.assertEqual(1, summary["repair_attempt_index"])
        written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(written_summary["repair_attempt"])
        tasks = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(tasks)
        self.assertTrue(all(task["tool_profile"] == "verify_necessary" for task in tasks))
        self.assertTrue(
            all(task["tool_subtype"] in {"verify_first", "scan_then_verify"} for task in tasks)
        )
        self.assertFalse(any(task["tool_subtype"] == "deceptive_scan" for task in tasks))

    def test_reference_repair_failed_preflight_writes_no_sha_and_no_live_outputs(self) -> None:
        output_root = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime/reference_repair_fail")
        output_path = output_root / "tasks.jsonl"
        summary_path = output_root / "task_pressure_preflight_summary.json"
        weak_summary = {
            "task_pressure_preflight_passes": True,
            "step_local_preflight_passes": False,
            "combined_preflight_passes": False,
        }
        with mock.patch("data.generate_tasks.summarize_toolized_profile", return_value=weak_summary):
            with self.assertRaises(RuntimeError):
                generate_tasks_file(
                    path=output_path,
                    count=20,
                    seed=11,
                    toolized=True,
                    toolized_profile="verify_necessary",
                    reference_repair=True,
                    repair_pool_count=40,
                    repair_selection_policy="minimal_preflight_margin_fix",
                    source_failed_attempt=Path(
                        "C:/Users/yaobc/Documents/Playground/logs/seed11_v3_reference_confirmation_20260430/reference_pack"
                    ),
                    repair_attempt_index=1,
                    preflight_summary_out=summary_path,
                )
        self.assertTrue(output_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertFalse(any(output_root.glob("*.sha256.txt")))
        self.assertFalse((output_root / "full.jsonl").exists())
        self.assertFalse((output_root / "selector_penalty.jsonl").exists())
        self.assertFalse((output_root / "voi_memory_v1.jsonl").exists())

    def test_validator_rejects_single_verify_oracle_task(self) -> None:
        task = {
            "task_id": "task-oracle",
            "answer": "A",
            "difficulty": "medium",
            "noise_level": 0.18,
            "ask_cost": -0.05,
            "max_steps": 4,
            "seed": 1,
            "toolized": True,
            "tool_profile": "verify_necessary",
            "tool_subtype": "verify_first",
            "clues": [
                {
                    "id": 1,
                    "scan_scores": {"A": -2, "B": 3, "C": 0},
                    "verify_scores": {"A": 3, "B": -2, "C": 0},
                    "scan_reliability": 0.45,
                    "verify_reliability": 0.95,
                    "scan_cost": -0.03,
                    "verify_cost": -0.08,
                    "scan_text": "",
                    "verify_text": "",
                },
                {
                    "id": 2,
                    "scan_scores": {"A": -2, "B": 3, "C": 0},
                    "verify_scores": {"A": -2, "B": 3, "C": 0},
                    "scan_reliability": 0.48,
                    "verify_reliability": 0.94,
                    "scan_cost": -0.03,
                    "verify_cost": -0.08,
                    "scan_text": "",
                    "verify_text": "",
                },
                {
                    "id": 3,
                    "scan_scores": {"A": -2, "C": 3, "B": 0},
                    "verify_scores": {"A": -2, "C": 3, "B": 0},
                    "scan_reliability": 0.5,
                    "verify_reliability": 0.93,
                    "scan_cost": -0.03,
                    "verify_cost": -0.08,
                    "scan_text": "",
                    "verify_text": "",
                },
            ],
        }
        validation = validate_toolized_task(task)
        self.assertFalse(validation["passes"])
        self.assertIn("positive_verify_count_below_two", validation["failures"])

    def test_validator_rejects_verify_first_when_template_choose_still_dominates(self) -> None:
        task = {
            "task_id": "task-template-dominates",
            "answer": "B",
            "difficulty": "medium",
            "noise_level": 0.18,
            "ask_cost": -0.05,
            "max_steps": 4,
            "seed": 11,
            "toolized": True,
            "tool_profile": "verify_necessary",
            "tool_subtype": "verify_first",
            "clues": _verify_first_tool_clues(
                answer="B",
                ask_cost=-0.05,
                clue_count=3,
                rng=random.Random(11),
            ),
        }
        validation = validate_toolized_task(task)
        self.assertFalse(validation["passes"])
        self.assertIn("initial_verify_not_better_than_template_choose", validation["failures"])
        self.assertTrue(validation["initial"]["template_choose_optimal"])

    def test_validator_accepts_deceptive_scan_subtype(self) -> None:
        deceptive_task = next(
            generated
            for generated in generate_tasks(
                count=12,
                seed=7,
                toolized=True,
                toolized_profile="verify_necessary_family_conflict_v2",
            )
            if generated["tool_subtype"] == "deceptive_scan"
        )
        validation = validate_toolized_task(deceptive_task)
        self.assertTrue(validation["passes"])
        self.assertEqual("deceptive_scan", validation["tool_subtype"])
        self.assertTrue(
            any(
                proxy_state["source_family"] == "scan"
                and proxy_state.get("best_verify_eval") is not None
                and proxy_state.get("best_choose_eval") is not None
                and proxy_state.get("best_scan_eval") is not None
                and float(proxy_state["best_verify_eval"]) - float(proxy_state["best_choose_eval"]) >= FAMILY_CONFLICT_DELTA
                and float(proxy_state["best_verify_eval"]) >= float(proxy_state["best_scan_eval"]) + FAMILY_CONFLICT_DELTA
                for proxy_state in validation["step_local"]["proxy_states"]
            )
        )

    def test_validator_accepts_v3_deceptive_scan_subtype(self) -> None:
        deceptive_task = next(
            generated
            for generated in generate_tasks(
                count=20,
                seed=7,
                toolized=True,
                toolized_profile="verify_necessary_aggressive_stress_v3",
            )
            if generated["tool_subtype"] == "deceptive_scan"
        )
        validation = validate_toolized_task(deceptive_task)
        self.assertTrue(validation["passes"])
        self.assertEqual(round(V3_FAMILY_CONFLICT_DELTA, 4), validation["step_local"]["family_conflict_delta_used"])


class ToolizedEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = generate_tasks(count=1, seed=7, toolized=True)[0]
        self.env = ToyTextMDP(self.task)

    def test_reset_exposes_scan_and_verify_actions(self) -> None:
        observation, _ = self.env.reset()
        options = set(observation["options"])
        first_clue_id = self.task["clues"][0]["id"]
        self.assertIn(f"ASK_SCAN_{first_clue_id}", options)
        self.assertIn(f"ASK_VERIFY_{first_clue_id}", options)

    def test_scan_consumes_clue_and_hides_verify_pair(self) -> None:
        observation, _ = self.env.reset()
        first_clue_id = self.task["clues"][0]["id"]
        next_observation, reward, done, info = self.env.step(f"ASK_SCAN_{first_clue_id}")
        self.assertFalse(done)
        self.assertEqual(float(self.task["clues"][0]["scan_cost"]), reward)
        self.assertEqual("scan", info["tool_family"])
        options = set(next_observation["options"])
        self.assertNotIn(f"ASK_SCAN_{first_clue_id}", options)
        self.assertNotIn(f"ASK_VERIFY_{first_clue_id}", options)

    def test_asked_clue_record_keeps_tool_metadata(self) -> None:
        self.env.reset()
        first_clue = self.task["clues"][0]
        observation, reward, done, _ = self.env.step(f"ASK_VERIFY_{first_clue['id']}")
        self.assertFalse(done)
        self.assertEqual(float(first_clue["verify_cost"]), reward)
        self.assertEqual(1, len(observation["asked_clues"]))
        asked_clue = observation["asked_clues"][0]
        self.assertEqual(first_clue["id"], asked_clue["id"])
        self.assertEqual("verify", asked_clue["tool_family"])
        self.assertEqual(float(first_clue["verify_reliability"]), asked_clue["reliability"])
        self.assertEqual(first_clue["verify_scores"], asked_clue["scores"])
        self.assertEqual(first_clue["verify_text"], asked_clue["text"])

    def test_belief_tracks_reliability_and_conflict(self) -> None:
        observation, _ = self.env.reset()
        first_clue = self.task["clues"][0]
        observation, _, _, _ = self.env.step(f"ASK_VERIFY_{first_clue['id']}")
        belief = belief_from_observation(observation)
        self.assertEqual(1, belief["evidence_count"])
        self.assertEqual(float(first_clue["verify_reliability"]), belief["mean_reliability"])
        self.assertGreaterEqual(belief["evidence_conflict"], 0.0)
        self.assertLessEqual(belief["evidence_conflict"], 1.0)
        self.assertTrue(all(action.startswith("ASK") for action in belief["remaining_asks"]))

    def test_template_candidates_cover_scan_verify_and_choose(self) -> None:
        observation, _ = self.env.reset()
        belief = belief_from_observation(observation)
        candidates = generate_candidates(observation, SelfModel(), candidate_limit=5)
        action_sets = {tuple(candidate["actions"]) for candidate in candidates}
        self.assertIn((f"CHOOSE_{belief['ranked_choices'][0]}",), action_sets)
        self.assertIn((f"CHOOSE_{belief['ranked_choices'][1]}",), action_sets)
        self.assertTrue(any(actions[:1] == ("ASK_SCAN_1",) and actions[-1] == "CHOOSE_BEST" for actions in action_sets))
        self.assertTrue(any(actions[:1] == ("ASK_VERIFY_1",) and actions[-1] == "CHOOSE_BEST" for actions in action_sets))

    def test_profile_tags_do_not_leak_into_observation_or_llm_prompt(self) -> None:
        task = generate_tasks(count=1, seed=7, toolized=True, toolized_profile="verify_necessary")[0]
        env = ToyTextMDP(task)
        observation, _ = env.reset()
        self.assertNotIn("tool_profile", observation["text"])
        self.assertNotIn("tool_subtype", observation["text"])

        client = _PromptCaptureClient()
        result = client.generate_candidates(
            task=task,
            observation=observation,
            self_model=SelfModel().to_dict(),
            memory_entries=[],
            candidate_limit=5,
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(client.captured_input)
        self.assertNotIn("tool_profile", client.captured_input)
        self.assertNotIn("tool_subtype", client.captured_input)
        self.assertNotIn("verify_necessary", client.captured_input)
        self.assertNotIn("scan_then_verify", client.captured_input)


class LegacyEnvironmentCompatibilityTests(unittest.TestCase):
    def test_legacy_tasks_still_expose_ask_clue_actions(self) -> None:
        task = generate_tasks(count=1, seed=7, toolized=False)[0]
        env = ToyTextMDP(task)
        observation, _ = env.reset()
        options = set(observation["options"])
        first_clue_id = task["clues"][0]["id"]
        self.assertIn(f"ASK_CLUE_{first_clue_id}", options)
        self.assertNotIn(f"ASK_SCAN_{first_clue_id}", options)
        self.assertNotIn(f"ASK_VERIFY_{first_clue_id}", options)


if __name__ == "__main__":
    unittest.main()
