from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from agent.counterfactual import belief_from_observation
from agent.loop import SelfExplanatoryAgent
from agent.memory import MemoryStore
from agent.voi_memory import (
    build_decision_lesson,
    build_decision_scope,
    evaluate_ask_gate,
    merge_voi_candidates,
)
from env.toytextmdp import ToyTextMDP

TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "test_voi_memory"
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _sample_task() -> dict[str, object]:
    return {
        "task_id": "task-0001",
        "answer": "A",
        "difficulty": "easy",
        "noise_level": 0.05,
        "ask_cost": -0.05,
        "max_steps": 4,
        "benchmark_family": "single_episode",
        "clues": [
            {
                "id": 1,
                "scores": {"A": 3, "B": -1, "C": -1},
                "reliability": 0.95,
                "text": "Clue 1: support(A:+3, B:-1, C:-1); reliability=0.95.",
            },
            {
                "id": 2,
                "scores": {"A": 2, "B": -1, "C": 0},
                "reliability": 0.93,
                "text": "Clue 2: support(A:+2, B:-1, C:+0); reliability=0.93.",
            },
            {
                "id": 3,
                "scores": {"A": 2, "B": 0, "C": -1},
                "reliability": 0.92,
                "text": "Clue 3: support(A:+2, B:+0, C:-1); reliability=0.92.",
            },
        ],
    }


def _toolized_sample_task() -> dict[str, object]:
    return {
        "task_id": "task-tool-0001",
        "answer": "A",
        "difficulty": "easy",
        "noise_level": 0.05,
        "ask_cost": -0.05,
        "max_steps": 4,
        "benchmark_family": "single_episode",
        "toolized": True,
        "clues": [
            {
                "id": 1,
                "scan_scores": {"A": 1, "B": 0, "C": 0},
                "verify_scores": {"A": 3, "B": -1, "C": -1},
                "scan_reliability": 0.6,
                "verify_reliability": 0.95,
                "scan_cost": -0.02,
                "verify_cost": -0.08,
                "scan_text": "Scan 1: weak support(A:+1); reliability=0.60.",
                "verify_text": "Verify 1: strong support(A:+3, B:-1, C:-1); reliability=0.95.",
            },
            {
                "id": 2,
                "scan_scores": {"A": 0, "B": 1, "C": 0},
                "verify_scores": {"A": 2, "B": -1, "C": 0},
                "scan_reliability": 0.58,
                "verify_reliability": 0.93,
                "scan_cost": -0.02,
                "verify_cost": -0.08,
                "scan_text": "Scan 2: slight support(B:+1); reliability=0.58.",
                "verify_text": "Verify 2: support(A:+2, B:-1); reliability=0.93.",
            },
        ],
    }


@contextmanager
def _workspace_tempdir() -> Path:
    path = TMP_ROOT / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class VoiMemoryHelperTests(unittest.TestCase):
    def test_merge_voi_candidates_preserves_template_coverage(self) -> None:
        llm_candidates = [
            {"name": "bad_choose", "actions": ["CHOOSE_A"], "category": "choose_now"},
            {"name": "ask_then_choose_a", "actions": ["ASK_CLUE_1", "CHOOSE_A"], "category": "ask_then_choose"},
        ]
        template_candidates = [
            {"name": "ask_best", "actions": ["ASK_CLUE_2", "CHOOSE_BEST"], "category": "ask_then_choose"},
            {"name": "choose_c", "actions": ["CHOOSE_C"], "category": "choose_now"},
            {"name": "choose_b", "actions": ["CHOOSE_B"], "category": "choose_alt"},
        ]
        merged = merge_voi_candidates(
            llm_candidates=llm_candidates,
            template_candidates=template_candidates,
        )
        action_sets = {tuple(candidate["actions"]) for candidate in merged}
        self.assertIn(("CHOOSE_A",), action_sets)
        self.assertIn(("CHOOSE_C",), action_sets)
        self.assertIn(("CHOOSE_B",), action_sets)
        self.assertIn(("ASK_CLUE_2", "CHOOSE_BEST"), action_sets)

    def test_build_decision_lesson_labels_and_buckets(self) -> None:
        task = _toolized_sample_task()
        belief = {
            "margin": 0.4,
            "remaining_asks": ["ASK_SCAN_2", "ASK_VERIFY_2"],
            "asked_count": 1,
            "mean_reliability": 0.72,
            "evidence_conflict": 0.4,
        }
        ask_regretful = build_decision_lesson(
            task=task,
            mode="voi_memory_v1",
            belief=belief,
            selected_action="ASK_VERIFY_2",
            selected_candidate_name="verify_again",
            ask_advantage=-0.08,
            ask_cost=-0.08,
            reward=-0.05,
            step_index=1,
        )
        choose_regretful = build_decision_lesson(
            task=task,
            mode="voi_memory_v1",
            belief=belief,
            selected_action="CHOOSE_A",
            selected_candidate_name="choose_now",
            ask_advantage=0.04,
            ask_cost=-0.08,
            reward=1.0,
            step_index=1,
            best_ask_family="scan",
        )
        self.assertIsNotNone(ask_regretful)
        self.assertEqual("ask_regretful", ask_regretful["lesson_type"])
        self.assertEqual("0p25_to_0p75", ask_regretful["margin_bucket"])
        self.assertEqual("2plus", ask_regretful["remaining_asks_bucket"])
        self.assertEqual("1", ask_regretful["asked_count_bucket"])
        self.assertEqual("verify", ask_regretful["tool_family"])
        self.assertEqual("0p55_to_0p80", ask_regretful["reliability_bucket"])
        self.assertEqual("0p34_to_0p67", ask_regretful["conflict_bucket"])
        self.assertIsNotNone(choose_regretful)
        self.assertEqual("choose_regretful", choose_regretful["lesson_type"])
        self.assertEqual("scan", choose_regretful["tool_family"])

    def test_evaluate_ask_gate_cases(self) -> None:
        candidates = [
            {"name": "ask", "actions": ["ASK_VERIFY_1"]},
            {"name": "choose", "actions": ["CHOOSE_A"]},
        ]
        strong_positive = evaluate_ask_gate(
            candidates=candidates,
            evaluations=[
                {"candidate_name": "ask", "score": 0.95},
                {"candidate_name": "choose", "score": 0.85},
            ],
            belief={"margin": 0.2, "mean_reliability": 0.5, "evidence_conflict": 0.0},
            confidence_threshold=1.25,
            ask_cost=-0.05,
            memory_retrieval={},
        )
        strong_negative = evaluate_ask_gate(
            candidates=candidates,
            evaluations=[
                {"candidate_name": "ask", "score": 0.70},
                {"candidate_name": "choose", "score": 0.85},
            ],
            belief={"margin": 1.5, "mean_reliability": 0.95, "evidence_conflict": 0.0},
            confidence_threshold=1.25,
            ask_cost=-0.05,
            memory_retrieval={},
        )
        borderline_allow = evaluate_ask_gate(
            candidates=candidates,
            evaluations=[
                {"candidate_name": "ask", "score": 0.84},
                {"candidate_name": "choose", "score": 0.85},
            ],
            belief={"margin": 0.1, "mean_reliability": 0.55, "evidence_conflict": 0.4},
            confidence_threshold=1.25,
            ask_cost=-0.05,
            memory_retrieval={
                "summary": {"regretful_ask": 0.6, "choose_helpful": 0.4},
                "by_family": {
                    "verify": {
                        "summary": {"helpful_ask": 0.8, "choose_regret": 0.6},
                    }
                },
            },
        )
        borderline_block = evaluate_ask_gate(
            candidates=candidates,
            evaluations=[
                {"candidate_name": "ask", "score": 0.84},
                {"candidate_name": "choose", "score": 0.85},
            ],
            belief={"margin": 1.2, "mean_reliability": 0.95, "evidence_conflict": 0.0},
            confidence_threshold=1.25,
            ask_cost=-0.05,
            memory_retrieval={
                "summary": {"helpful_ask": 0.8, "choose_regret": 0.6},
                "by_family": {
                    "verify": {
                        "summary": {"regretful_ask": 0.8, "choose_helpful": 0.6},
                    }
                },
            },
        )
        self.assertTrue(strong_positive["allowed"])
        self.assertEqual("strong_positive_voi", strong_positive["reason"])
        self.assertFalse(strong_negative["allowed"])
        self.assertEqual("strong_negative_voi", strong_negative["reason"])
        self.assertTrue(borderline_allow["allowed"])
        self.assertFalse(borderline_block["allowed"])
        self.assertEqual("verify", borderline_allow["inputs"]["best_ask_family"])
        self.assertGreater(borderline_allow["inputs"]["uncertainty"], borderline_allow["inputs"]["base_uncertainty"])


class DecisionLessonStoreTests(unittest.TestCase):
    def test_upsert_reinforces_instead_of_duplicating(self) -> None:
        with _workspace_tempdir() as tmpdir:
            store = MemoryStore(tmpdir / "memory.jsonl")
            task = _toolized_sample_task()
            belief = {
                "margin": 0.4,
                "remaining_asks": ["ASK_VERIFY_2"],
                "asked_count": 1,
                "mean_reliability": 0.72,
                "evidence_conflict": 0.4,
            }
            lesson = build_decision_lesson(
                task=task,
                mode="voi_memory_v1",
                belief=belief,
                selected_action="ASK_VERIFY_2",
                selected_candidate_name="verify_again",
                ask_advantage=-0.08,
                ask_cost=-0.08,
                reward=-0.05,
                step_index=1,
            )
            self.assertIsNotNone(lesson)
            first = store.upsert_decision_lesson(lesson)
            self.assertIsNotNone(first)

            lesson["strength"] = 0.9
            second = store.upsert_decision_lesson(lesson)
            self.assertIsNotNone(second)
            entries = store.read_all()
            decision_lessons = [entry for entry in entries if entry.get("type") == "decision_lesson"]
            self.assertEqual(1, len(decision_lessons))
            self.assertEqual(2, int(decision_lessons[0]["reinforcement_count"]))
            self.assertGreater(float(decision_lessons[0]["strength"]), 0.5)

    def test_pruning_keeps_only_top_40_per_type(self) -> None:
        with _workspace_tempdir() as tmpdir:
            store = MemoryStore(tmpdir / "memory.jsonl")
            for index in range(45):
                entry = {
                    "type": "decision_lesson",
                    "lesson_type": "ask_regretful",
                    "difficulty": "easy",
                    "noise_bucket": f"{0.01 * index:.2f}",
                    "margin_bucket": "0p25_to_0p75",
                    "remaining_asks_bucket": "1",
                    "asked_count_bucket": "0",
                    "tool_family": "scan" if index % 2 == 0 else "verify",
                    "reliability_bucket": "0p55_to_0p80",
                    "conflict_bucket": "none",
                    "strength": min(1.0, 0.2 + index / 50.0),
                    "task_tags": [],
                    "provenance": {"task_id": f"task-{index:04d}"},
                    "content": f"lesson {index}",
                }
                store.upsert_decision_lesson(entry)
            lessons = [
                entry
                for entry in store.read_all()
                if entry.get("type") == "decision_lesson"
                and entry.get("lesson_type") == "ask_regretful"
            ]
            self.assertEqual(40, len(lessons))

    def test_typed_retrieval_prefers_close_scope_matches(self) -> None:
        with _workspace_tempdir() as tmpdir:
            store = MemoryStore(tmpdir / "memory.jsonl")
            store.append(
                {
                    "type": "decision_lesson",
                    "lesson_type": "ask_helpful",
                    "difficulty": "easy",
                    "noise_bucket": "0.05",
                    "margin_bucket": "0p25_to_0p75",
                    "remaining_asks_bucket": "1",
                    "asked_count_bucket": "1",
                    "tool_family": "scan",
                    "reliability_bucket": "0p55_to_0p80",
                    "conflict_bucket": "none",
                    "strength": 0.9,
                    "last_seen_version": 1,
                    "task_tags": [],
                    "provenance": {"task_id": "task-0001"},
                    "content": "exact match",
                }
            )
            store.append(
                {
                    "type": "decision_lesson",
                    "lesson_type": "ask_helpful",
                    "difficulty": "hard",
                    "noise_bucket": "0.32",
                    "margin_bucket": "ge_1p50",
                    "remaining_asks_bucket": "0",
                    "asked_count_bucket": "2plus",
                    "tool_family": "verify",
                    "reliability_bucket": "ge_0p80",
                    "conflict_bucket": "ge_0p67",
                    "strength": 0.95,
                    "last_seen_version": 2,
                    "task_tags": [],
                    "provenance": {"task_id": "task-0002"},
                    "content": "mismatch",
                }
            )
            store.append(
                {
                    "type": "decision_lesson",
                    "lesson_type": "ask_regretful",
                    "difficulty": "easy",
                    "noise_bucket": "0.05",
                    "margin_bucket": "0p25_to_0p75",
                    "remaining_asks_bucket": "1",
                    "asked_count_bucket": "1",
                    "tool_family": "scan",
                    "reliability_bucket": "0p55_to_0p80",
                    "conflict_bucket": "none",
                    "strength": 0.8,
                    "last_seen_version": 3,
                    "task_tags": [],
                    "provenance": {"task_id": "task-0003"},
                    "content": "caution exact match",
                }
            )
            scope = build_decision_scope(
                task=_toolized_sample_task(),
                belief={
                    "margin": 0.4,
                    "remaining_asks": ["ASK_SCAN_2"],
                    "asked_count": 1,
                    "mean_reliability": 0.72,
                    "evidence_conflict": 0.0,
                },
                tool_family="scan",
            )
            retrieved = store.search_decision_lessons(query_scope=scope)
            self.assertEqual("exact match", retrieved["positive"][0]["content"])
            self.assertEqual("caution exact match", retrieved["caution"][0]["content"])
            self.assertGreater(retrieved["summary"]["helpful_ask"], 0.0)
            self.assertGreater(retrieved["summary"]["regretful_ask"], 0.0)
            self.assertEqual("scan", retrieved["query_tool_family"])
            self.assertIn("scan", retrieved["positive_tool_families"])


class AgentIntegrationSmokeTests(unittest.TestCase):
    def test_voi_mode_unions_llm_and_template_candidates(self) -> None:
        with _workspace_tempdir() as tmpdir:
            memory_path = tmpdir / "voi_memory_union.jsonl"
            audit_path = tmpdir / "voi_memory_union_audit.jsonl"
            agent = SelfExplanatoryAgent(
                mode="voi_memory_v1",
                memory_path=memory_path,
                audit_log_path=audit_path,
                seed=7,
            )
            agent.llm.generate_candidates = lambda **_: [  # type: ignore[method-assign]
                {
                    "name": "immediate_choose_a",
                    "actions": ["CHOOSE_A"],
                    "category": "choose_now",
                }
            ]
            agent.llm.generate_reflection = lambda **_: None  # type: ignore[method-assign]
            records = agent.run_task(ToyTextMDP(_sample_task()))
            first_step = next(record for record in records if record.get("record_type") == "step")
            action_sets = {tuple(candidate["actions"]) for candidate in first_step["candidates"]}
            self.assertEqual("openai_responses_plus_template", first_step["candidate_backend"])
            self.assertTrue(first_step["template_candidates"])
            self.assertTrue(first_step["llm_candidates"])
            self.assertTrue(first_step["selector_candidates"])
            self.assertTrue(first_step["selector_evaluations"])
            self.assertIn("candidate_count_by_family_before_normalization", first_step)
            self.assertIn("candidate_count_by_family_after_normalization", first_step)
            self.assertIn("best_scan_surface_shape_before", first_step)
            self.assertIn("best_verify_surface_shape_before", first_step)
            self.assertIn("best_scan_surface_shape_after", first_step)
            self.assertIn("best_verify_surface_shape_after", first_step)
            self.assertIn("verify_candidate_exposed", first_step)
            self.assertIn("best_verify_missing_reason", first_step)
            self.assertGreaterEqual(
                int(first_step["candidate_count_by_family_after_normalization"].get("clue", 0)),
                1,
            )
            self.assertIn(("CHOOSE_A",), action_sets)
            self.assertIn(("CHOOSE_C",), action_sets)
            self.assertIn(("CHOOSE_B",), action_sets)
            self.assertTrue(any(actions[-1] == "CHOOSE_BEST" for actions in action_sets))

    def test_voi_mode_uses_gate_and_skips_legacy_reflection_write(self) -> None:
        with _workspace_tempdir() as tmpdir:
            memory_path = tmpdir / "voi_memory.jsonl"
            audit_path = tmpdir / "voi_audit.jsonl"
            agent = SelfExplanatoryAgent(
                mode="voi_memory_v1",
                memory_path=memory_path,
                audit_log_path=audit_path,
                seed=7,
            )
            agent.llm.generate_candidates = lambda **_: None  # type: ignore[method-assign]
            agent.llm.generate_reflection = lambda **_: None  # type: ignore[method-assign]
            records = agent.run_task(ToyTextMDP(_sample_task()))
            step_records = [record for record in records if record.get("record_type") == "step"]
            episode_record = next(record for record in records if record.get("record_type") == "episode")
            self.assertTrue(step_records)
            self.assertIsNotNone(step_records[0]["ask_gate_allowed"])
            self.assertIsNotNone(step_records[0]["retrieved_decision_lessons"])
            self.assertIsNone(episode_record["reflection_written"])
            self.assertFalse(episode_record["reflection_admitted"])
            self.assertEqual({}, episode_record["self_model_delta_applied"])

    def test_toolized_voi_mode_uses_family_scoped_retrieval_and_writes_tool_lesson(self) -> None:
        with _workspace_tempdir() as tmpdir:
            memory_path = tmpdir / "tool_voi_memory.jsonl"
            audit_path = tmpdir / "tool_voi_audit.jsonl"
            task = _toolized_sample_task()
            seeded_env = ToyTextMDP(task)
            initial_observation, _ = seeded_env.reset()
            initial_belief = belief_from_observation(initial_observation)
            store = MemoryStore(memory_path)
            for lesson_type, family, content in (
                ("ask_helpful", "scan", "scan helpful"),
                ("ask_regretful", "verify", "verify caution"),
            ):
                scope = build_decision_scope(task=task, belief=initial_belief, tool_family=family)
                store.append(
                    {
                        "type": "decision_lesson",
                        "lesson_type": lesson_type,
                        "difficulty": scope["difficulty"],
                        "noise_bucket": scope["noise_bucket"],
                        "margin_bucket": scope["margin_bucket"],
                        "remaining_asks_bucket": scope["remaining_asks_bucket"],
                        "asked_count_bucket": scope["asked_count_bucket"],
                        "tool_family": scope["tool_family"],
                        "reliability_bucket": scope["reliability_bucket"],
                        "conflict_bucket": scope["conflict_bucket"],
                        "strength": 0.9,
                        "last_seen_version": 1,
                        "task_tags": [],
                        "provenance": {"task_id": task["task_id"]},
                        "content": content,
                    }
                )

            agent = SelfExplanatoryAgent(
                mode="voi_memory_v1",
                memory_path=memory_path,
                audit_log_path=audit_path,
                seed=7,
            )
            agent.llm.generate_candidates = lambda **_: None  # type: ignore[method-assign]
            agent.llm.generate_reflection = lambda **_: None  # type: ignore[method-assign]
            records = agent.run_task(ToyTextMDP(task))
            step_records = [record for record in records if record.get("record_type") == "step"]
            first_step = step_records[0]
            retrieved = first_step["retrieved_decision_lessons"]
            self.assertIn("scan", retrieved["by_family"])
            self.assertIn("verify", retrieved["by_family"])
            self.assertEqual("scan", retrieved["by_family"]["scan"]["query_tool_family"])
            self.assertEqual("verify", retrieved["by_family"]["verify"]["query_tool_family"])
            self.assertEqual(["scan"], retrieved["by_family"]["scan"]["positive_tool_families"])
            self.assertEqual(["verify"], retrieved["by_family"]["verify"]["caution_tool_families"])
            written = next(
                (
                    record["written_decision_lesson"]
                    for record in step_records
                    if record.get("written_decision_lesson") is not None
                ),
                None,
            )
            self.assertIsNotNone(written)
            self.assertIn(written["tool_family"], {"scan", "verify", "none"})
            self.assertIn(
                written["reliability_bucket"],
                {"lt_0p55", "0p55_to_0p80", "ge_0p80"},
            )
            self.assertIn(
                written["conflict_bucket"],
                {"none", "lt_0p34", "0p34_to_0p67", "ge_0p67"},
            )

    def test_voi_coverage_only_unions_candidates_without_gate(self) -> None:
        with _workspace_tempdir() as tmpdir:
            memory_path = tmpdir / "voi_coverage_only.jsonl"
            audit_path = tmpdir / "voi_coverage_only_audit.jsonl"
            agent = SelfExplanatoryAgent(
                mode="voi_coverage_only",
                memory_path=memory_path,
                audit_log_path=audit_path,
                seed=7,
            )
            agent.llm.generate_candidates = lambda **_: [  # type: ignore[method-assign]
                {
                    "name": "immediate_choose_a",
                    "actions": ["CHOOSE_A"],
                    "category": "choose_now",
                }
            ]
            agent.llm.generate_reflection = lambda **_: None  # type: ignore[method-assign]
            records = agent.run_task(ToyTextMDP(_sample_task()))
            first_step = next(record for record in records if record.get("record_type") == "step")
            action_sets = {tuple(candidate["actions"]) for candidate in first_step["candidates"]}
            episode_record = next(record for record in records if record.get("record_type") == "episode")
            self.assertEqual("openai_responses_plus_template", first_step["candidate_backend"])
            self.assertIn(("CHOOSE_A",), action_sets)
            self.assertIn(("CHOOSE_C",), action_sets)
            self.assertIsNone(first_step["ask_gate_allowed"])
            self.assertIsNone(first_step["retrieved_decision_lessons"])
            self.assertIsNone(episode_record["reflection_written"])
            self.assertFalse(episode_record["reflection_admitted"])

    def test_voi_gate_only_uses_gate_without_decision_lessons(self) -> None:
        with _workspace_tempdir() as tmpdir:
            memory_path = tmpdir / "voi_gate_only.jsonl"
            audit_path = tmpdir / "voi_gate_only_audit.jsonl"
            agent = SelfExplanatoryAgent(
                mode="voi_gate_only",
                memory_path=memory_path,
                audit_log_path=audit_path,
                seed=7,
            )
            agent.llm.generate_candidates = lambda **_: None  # type: ignore[method-assign]
            agent.llm.generate_reflection = lambda **_: None  # type: ignore[method-assign]
            records = agent.run_task(ToyTextMDP(_sample_task()))
            first_step = next(record for record in records if record.get("record_type") == "step")
            episode_record = next(record for record in records if record.get("record_type") == "episode")
            self.assertIsNotNone(first_step["ask_gate_allowed"])
            self.assertIsNone(first_step["retrieved_decision_lessons"])
            self.assertIsNone(first_step["written_decision_lesson"])
            self.assertIsNone(episode_record["reflection_written"])
            self.assertFalse(episode_record["reflection_admitted"])

    def test_full_mode_does_not_emit_gate_fields(self) -> None:
        with _workspace_tempdir() as tmpdir:
            memory_path = tmpdir / "full_memory.jsonl"
            audit_path = tmpdir / "full_audit.jsonl"
            agent = SelfExplanatoryAgent(
                mode="full",
                memory_path=memory_path,
                audit_log_path=audit_path,
                seed=7,
            )
            agent.llm.generate_candidates = lambda **_: None  # type: ignore[method-assign]
            agent.llm.generate_reflection = lambda **_: None  # type: ignore[method-assign]
            records = agent.run_task(ToyTextMDP(_sample_task()))
            step_records = [record for record in records if record.get("record_type") == "step"]
            self.assertTrue(step_records)
            self.assertIsNone(step_records[0]["ask_gate_allowed"])
            self.assertIsNone(step_records[0]["retrieved_decision_lessons"])


if __name__ == "__main__":
    unittest.main()
