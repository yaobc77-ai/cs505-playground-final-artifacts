from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from agent.candidate_coverage import (
    best_family_surface_shape,
    build_candidate_layer_bundle,
    build_verify_lineage_survival,
    infer_best_verify_missing_reason,
    infer_verify_candidate_exposed,
)
from agent.commitment import build_commitment_entry, build_commitment_record
from agent.counterfactual import belief_from_observation, evaluate_candidate, generate_candidates
from agent.episodic_memory import compose_subject_memory
from agent.governance import Governance
from agent.lack_engine import build_lack_snapshot
from agent.llm import OpenAIResponsesClient
from agent.memory import MemoryStore
from agent.reflection import build_reflection
from agent.self_model import SelfModel
from agent.selector import select_candidate
from agent.subjective_self import build_thin_self_model
from agent.symbolic_other import build_symbolic_other
from agent.voi_memory import (
    action_tool_family,
    build_decision_lesson,
    build_decision_scope,
    evaluate_ask_gate,
    merge_voi_candidates,
)

SUBJECT_MODES = {
    "subject_v0",
    "subject_no_other",
    "subject_no_commitment",
    "subject_no_memory_recompose",
}
VOI_FAMILY_MODES = {
    "voi_memory_v1",
    "voi_gate_only",
    "voi_coverage_only",
}


def _task_tags(task: dict[str, Any], mode: str) -> list[str]:
    tags = [
        f"difficulty:{task.get('difficulty', 'unknown')}",
        f"noise:{task.get('noise_level', 0.0)}",
        f"mode:{mode}",
    ]
    if task.get("tool_profile"):
        tags.append(f"tool_profile:{task['tool_profile']}")
    if task.get("tool_subtype"):
        tags.append(f"tool_subtype:{task['tool_subtype']}")
    if task.get("benchmark_family"):
        tags.append(f"benchmark:{task['benchmark_family']}")
    if task.get("arc_id"):
        tags.append(f"arc:{task['arc_id']}")
    if task.get("episode_index") is not None:
        tags.append(f"episode:{task['episode_index']}")
    return tags


def _empty_subject_memory(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "arc_id": str(task.get("arc_id", "")),
        "identity_context": dict(task.get("identity_context", {})),
        "supporting_lessons": [],
        "conflicting_lessons": [],
        "prior_commitments": [],
        "unresolved_contradictions": [],
        "supporting_count": 0,
        "conflicting_count": 0,
        "prior_commitment_count": 0,
        "unresolved_count": 0,
        "arc_memory_count": 0,
        "commitment_alignment_count": 0,
    }


def _neutral_symbolic_other() -> dict[str, Any]:
    return {
        "expected_reason_type": "none",
        "recognized_actions": [],
        "action_bias": {"ASK": 0.0, "CHOOSE": 0.0},
        "normative_pressure": 0.0,
        "consistency_pressure": 0.0,
        "signals": [],
        "recognition_prompt": "",
    }


class SelfExplanatoryAgent:
    def __init__(
        self,
        *,
        mode: str,
        memory_path: Path | str,
        audit_log_path: Path | str,
        seed: int = 7,
        selector_config: dict[str, Any] | None = None,
    ) -> None:
        self.mode = mode
        self.memory = MemoryStore(memory_path)
        self.governance = Governance(audit_log_path)
        self.self_model = SelfModel()
        self.llm = OpenAIResponsesClient()
        self.random = random.Random(seed)
        self.selector_config = dict(selector_config or {})

    @property
    def subject_mode_enabled(self) -> bool:
        return self.mode in SUBJECT_MODES

    @property
    def voi_memory_mode_enabled(self) -> bool:
        return self.mode == "voi_memory_v1"

    @property
    def voi_family_enabled(self) -> bool:
        return self.mode in VOI_FAMILY_MODES

    @property
    def voi_candidate_union_enabled(self) -> bool:
        return self.mode in VOI_FAMILY_MODES

    @property
    def voi_gate_enabled(self) -> bool:
        return self.mode in {"voi_memory_v1", "voi_gate_only"}

    @property
    def voi_decision_lesson_enabled(self) -> bool:
        return self.mode == "voi_memory_v1"

    def _retrieve_memory(self, task: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, Any]]:
        if self.mode == "no_reflection":
            return []

        belief = belief_from_observation(observation)
        query_terms = [
            task.get("difficulty", ""),
            belief["top_choice"],
            "ask" if belief["remaining_asks"] else "choose",
        ]
        task_tags = _task_tags(task, self.mode)
        if task.get("arc_id"):
            query_terms.append(str(task["arc_id"]))
        if (task.get("identity_context") or {}).get("style"):
            query_terms.append(str(task["identity_context"]["style"]))
        return self.memory.search(query_terms=query_terms, task_tags=task_tags, limit=8 if self.subject_mode_enabled else 5)

    def _retrieve_decision_lessons(
        self,
        *,
        task: dict[str, Any],
        belief: dict[str, Any],
    ) -> dict[str, Any]:
        remaining_asks = list(belief.get("remaining_asks", []))
        families = sorted(
            {
                family
                for family in (action_tool_family(action) for action in remaining_asks)
                if family in {"scan", "verify", "clue"}
            }
        )
        if not families:
            families = ["clue"] if not task.get("toolized") else []

        by_family: dict[str, dict[str, Any]] = {}
        combined_positive: list[dict[str, Any]] = []
        combined_caution: list[dict[str, Any]] = []
        seen_positive_ids: set[str] = set()
        seen_caution_ids: set[str] = set()

        for family in families:
            query_scope = build_decision_scope(task=task, belief=belief, tool_family=family)
            retrieved = self.memory.search_decision_lessons(query_scope=query_scope)
            by_family[family] = retrieved
            for entry in retrieved.get("positive", []):
                memory_id = str(entry.get("memory_id", ""))
                if memory_id and memory_id in seen_positive_ids:
                    continue
                if memory_id:
                    seen_positive_ids.add(memory_id)
                combined_positive.append(entry)
            for entry in retrieved.get("caution", []):
                memory_id = str(entry.get("memory_id", ""))
                if memory_id and memory_id in seen_caution_ids:
                    continue
                if memory_id:
                    seen_caution_ids.add(memory_id)
                combined_caution.append(entry)

        overall_summary = {
            "helpful_ask": 0.0,
            "choose_regret": 0.0,
            "regretful_ask": 0.0,
            "choose_helpful": 0.0,
        }
        for family_retrieval in by_family.values():
            summary = dict(family_retrieval.get("summary", {}))
            for key in overall_summary:
                overall_summary[key] = min(1.0, overall_summary[key] + float(summary.get(key, 0.0)))

        return {
            "query_scope": build_decision_scope(task=task, belief=belief),
            "by_family": by_family,
            "positive": combined_positive,
            "caution": combined_caution,
            "summary": {
                key: round(value, 4)
                for key, value in overall_summary.items()
            },
        }

    def _evaluate_candidates(
        self,
        env: Any,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        evaluations = [evaluate_candidate(env, candidate) for candidate in candidates]

        if self.mode == "no_cf_eval":
            for evaluation in evaluations:
                evaluation["score"] = 0.0
                evaluation["cf_mode"] = "disabled"
        elif self.mode == "permute":
            shuffled_scores = [evaluation["score"] for evaluation in evaluations]
            self.random.shuffle(shuffled_scores)
            for evaluation, shuffled_score in zip(evaluations, shuffled_scores, strict=True):
                evaluation["score"] = shuffled_score
                evaluation["cf_mode"] = "permuted"
        else:
            for evaluation in evaluations:
                evaluation["cf_mode"] = "standard"
        return evaluations

    def run_task(self, env: Any) -> list[dict[str, Any]]:
        observation, _ = env.reset()
        done = False
        total_reward = 0.0
        step_records: list[dict[str, Any]] = []
        retrieved_memory_ids: list[str] = []
        self_model_before_episode = self.self_model.to_dict()

        last_selection: dict[str, Any] | None = None
        last_thin_self_model: dict[str, float] | None = None
        last_subject_memory: dict[str, Any] = _empty_subject_memory(env.task_config)
        last_symbolic_other: dict[str, Any] = _neutral_symbolic_other()
        last_lack_snapshot: dict[str, Any] = {}
        last_arc_state: dict[str, Any] = {
            "benchmark_family": str(env.task_config.get("benchmark_family", "")),
            "arc_id": env.task_config.get("arc_id"),
            "episode_index": env.task_config.get("episode_index"),
            "prior_commitment_count": 0,
            "arc_memory_count": 0,
            "commitment_revisit_required": False,
        }

        while not done and env.state["step"] < env.max_steps:
            belief = belief_from_observation(observation)
            legacy_memory_entries: list[dict[str, Any]] = []
            decision_lesson_retrieval = {
                "query_scope": {},
                "by_family": {},
                "positive": [],
                "caution": [],
                "summary": {
                    "helpful_ask": 0.0,
                    "choose_regret": 0.0,
                    "regretful_ask": 0.0,
                    "choose_helpful": 0.0,
                },
            }
            if self.voi_decision_lesson_enabled:
                decision_lesson_retrieval = self._retrieve_decision_lessons(
                    task=env.task_config,
                    belief=belief,
                )
                retrieved_memory_ids = [
                    str(entry.get("memory_id", ""))
                    for entry in decision_lesson_retrieval.get("positive", [])
                    + decision_lesson_retrieval.get("caution", [])
                    if entry.get("memory_id")
                ]
            elif not self.voi_family_enabled:
                legacy_memory_entries = self._retrieve_memory(env.task_config, observation)
                retrieved_memory_ids = [entry.get("memory_id", "") for entry in legacy_memory_entries]
            current_model = self.self_model if self.mode != "no_self_model" else SelfModel()

            subject_memory = _empty_subject_memory(env.task_config)
            thin_self_model_obj = None
            symbolic_other = _neutral_symbolic_other()
            lack_snapshot: dict[str, Any] = {}
            arc_state = {
                "benchmark_family": str(env.task_config.get("benchmark_family", "")),
                "arc_id": env.task_config.get("arc_id"),
                "episode_index": env.task_config.get("episode_index"),
                "prior_commitment_count": 0,
                "arc_memory_count": 0,
                "commitment_revisit_required": bool(
                    (env.task_config.get("commitment_revisit") or {}).get("required")
                ),
            }
            subject_context = None

            if self.subject_mode_enabled:
                if self.mode != "subject_no_memory_recompose":
                    subject_memory = compose_subject_memory(
                        legacy_memory_entries,
                        task=env.task_config,
                        belief=belief,
                    )
                thin_self_model_obj = build_thin_self_model(
                    task=env.task_config,
                    belief=belief,
                    composed_memory=subject_memory,
                )
                symbolic_other = (
                    _neutral_symbolic_other()
                    if self.mode == "subject_no_other"
                    else build_symbolic_other(
                        task=env.task_config,
                        belief=belief,
                        composed_memory=subject_memory,
                    )
                )
                lack_snapshot = build_lack_snapshot(
                    task=env.task_config,
                    belief=belief,
                    thin_self_model=thin_self_model_obj,
                    symbolic_other=symbolic_other,
                    composed_memory=subject_memory,
                )
                arc_state.update(
                    {
                        "prior_commitment_count": int(subject_memory.get("prior_commitment_count", 0)),
                        "arc_memory_count": int(subject_memory.get("arc_memory_count", 0)),
                    }
                )
                subject_context = {
                    "thin_self_model_obj": thin_self_model_obj,
                    "composed_memory": subject_memory,
                    "symbolic_other": symbolic_other,
                    "lack_snapshot": lack_snapshot,
                }

            candidate_backend = "template"
            candidate_limit = 5
            template_candidates = generate_candidates(observation, current_model, candidate_limit=candidate_limit)
            llm_candidates = self.llm.generate_candidates(
                task=env.task_config,
                observation=observation,
                self_model=current_model.to_dict(),
                memory_entries=[] if self.voi_family_enabled else legacy_memory_entries,
                candidate_limit=candidate_limit,
                subject_context={
                    "subject_mode_enabled": self.subject_mode_enabled,
                    "subject_memory": subject_memory,
                    "symbolic_other": symbolic_other,
                    "lack_snapshot": lack_snapshot,
                }
                if self.subject_mode_enabled
                else None,
            )
            if llm_candidates and self.voi_candidate_union_enabled:
                raw_candidates = merge_voi_candidates(
                    llm_candidates=llm_candidates,
                    template_candidates=template_candidates,
                )
                candidate_backend = "openai_responses_plus_template"
            elif llm_candidates:
                raw_candidates = llm_candidates
                candidate_backend = "openai_responses"
            else:
                raw_candidates = template_candidates
                if self.llm.configured:
                    candidate_backend = "template_fallback"
            candidate_layer = build_candidate_layer_bundle(
                base_candidates=raw_candidates,
                available_actions=[str(action) for action in observation.get("options", [])],
                template_candidates=template_candidates,
            )
            candidates_before_normalization = list(candidate_layer["candidates_before_normalization"])
            evaluations_before_normalization = self._evaluate_candidates(env, candidates_before_normalization)
            best_scan_surface_shape_before = best_family_surface_shape(
                candidates_before_normalization,
                evaluations_before_normalization,
                "scan",
            )
            best_verify_surface_shape_before = best_family_surface_shape(
                candidates_before_normalization,
                evaluations_before_normalization,
                "verify",
            )
            candidates = list(candidate_layer["candidates_after_normalization"])
            evaluations = self._evaluate_candidates(env, candidates)
            best_scan_surface_shape_after = best_family_surface_shape(candidates, evaluations, "scan")
            best_verify_surface_shape_after = best_family_surface_shape(candidates, evaluations, "verify")
            ask_gate = {
                "allowed": True,
                "score": None,
                "reason": "disabled",
                "inputs": {},
            }
            selector_candidates = candidates
            selector_evaluations = evaluations
            pre_selector_filter_applied = False
            pre_selector_filtered_pairs_empty = False
            pre_selector_original_candidate_count = len(candidates)
            pre_selector_filtered_candidate_count = len(selector_candidates)
            selector_memory_entries = [] if self.voi_family_enabled else legacy_memory_entries
            if self.voi_gate_enabled:
                ask_gate = evaluate_ask_gate(
                    candidates=candidates,
                    evaluations=evaluations,
                    belief=belief,
                    confidence_threshold=current_model.confidence_threshold,
                    ask_cost=env.ask_cost,
                    memory_retrieval=(
                        decision_lesson_retrieval
                        if self.voi_decision_lesson_enabled
                        else {}
                    ),
                )
                if not ask_gate["allowed"]:
                    filtered_pairs = [
                        (candidate, evaluation)
                        for candidate, evaluation in zip(candidates, evaluations, strict=True)
                        if not str(candidate.get("actions", [""])[0]).startswith("ASK")
                    ]
                    if filtered_pairs:
                        pre_selector_filter_applied = True
                        selector_candidates = [candidate for candidate, _ in filtered_pairs]
                        selector_evaluations = [evaluation for _, evaluation in filtered_pairs]
                        pre_selector_filtered_candidate_count = len(selector_candidates)
                    else:
                        pre_selector_filtered_pairs_empty = True
                        ask_gate["reason"] = "blocked_fallback_original"
            else:
                pre_selector_filtered_candidate_count = len(selector_candidates)
            verify_candidate_exposed = infer_verify_candidate_exposed(selector_candidates)
            best_verify_missing_reason = infer_best_verify_missing_reason(
                remaining_verify_actions=[str(action) for action in belief.get("remaining_verify_actions", [])],
                candidates_before_normalization=candidates_before_normalization,
                candidates_after_normalization=candidates,
                selector_candidates=selector_candidates,
                selector_evaluations=selector_evaluations,
            )
            verify_lineage_audit = build_verify_lineage_survival(
                remaining_verify_actions=[str(action) for action in belief.get("remaining_verify_actions", [])],
                candidate_backend=candidate_backend,
                candidate_limit=candidate_limit,
                template_candidates=template_candidates,
                llm_candidates=llm_candidates,
                source_candidates=list(candidate_layer.get("source_candidates", [])),
                normalized_candidates_pre_dedup=list(candidate_layer.get("normalized_candidates_pre_dedup", [])),
                deduped_candidates=list(candidate_layer.get("candidates_after_dedup_before_exposure", [])),
                final_candidates=candidates,
                selector_candidates=selector_candidates,
                ask_gate_allowed=ask_gate.get("allowed") if self.voi_gate_enabled else None,
                pre_selector_filter_evidence=(
                    self.voi_gate_enabled
                    and (ask_gate.get("allowed") is False)
                    and (pre_selector_filter_applied or pre_selector_filtered_pairs_empty)
                ),
            )
            selection = select_candidate(
                observation=observation,
                candidates=selector_candidates,
                evaluations=selector_evaluations,
                self_model=current_model,
                memory_entries=selector_memory_entries,
                selector_mode="selector_penalty" if self.mode == "selector_penalty" else "default",
                selector_config=self.selector_config,
                subject_context=subject_context,
            )
            action = selection["candidate"]["actions"][0]
            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            written_decision_lesson: dict[str, Any] | None = None
            if self.voi_decision_lesson_enabled:
                lesson_entry = build_decision_lesson(
                    task=env.task_config,
                    mode=self.mode,
                    belief=belief,
                    selected_action=action,
                    selected_candidate_name=str(selection["candidate"].get("name", "")),
                    ask_advantage=ask_gate.get("inputs", {}).get("ask_advantage"),
                    ask_cost=env.ask_cost,
                    reward=reward,
                    step_index=int(env.state["step"]),
                    best_ask_family=str(ask_gate.get("inputs", {}).get("best_ask_family") or ""),
                )
                if lesson_entry:
                    written_decision_lesson = self.memory.upsert_decision_lesson(lesson_entry)

            step_record = {
                "record_type": "step",
                "mode": self.mode,
                "task_id": env.task_id,
                "step": env.state["step"],
                "action": action,
                "reward": reward,
                "done": done,
                "observation_text": observation["text"],
                "belief": belief,
                "self_model": current_model.to_dict(),
                "thin_self_model": thin_self_model_obj.to_dict() if thin_self_model_obj else None,
                "subject_memory": subject_memory if self.subject_mode_enabled else None,
                "symbolic_other": symbolic_other if self.subject_mode_enabled else None,
                "lack_snapshot": lack_snapshot if self.subject_mode_enabled else None,
                "recognition_judgment": selection.get("recognition_judgment") if self.subject_mode_enabled else None,
                "arc_state": dict(arc_state) if self.subject_mode_enabled else None,
                "retrieved_memory_ids": retrieved_memory_ids,
                "retrieved_decision_lessons": (
                    decision_lesson_retrieval if self.voi_decision_lesson_enabled else None
                ),
                "candidates_before_normalization": candidates_before_normalization,
                "evaluations_before_normalization": evaluations_before_normalization,
                "template_candidates": template_candidates,
                "llm_candidates": llm_candidates,
                "source_candidates": list(candidate_layer.get("source_candidates", [])),
                "candidate_limit": candidate_limit,
                "candidate_normalization_trace": candidate_layer["candidate_normalization_trace"],
                "candidate_exposure_trace": candidate_layer["candidate_exposure_trace"],
                "candidate_count_by_family_before_normalization": candidate_layer[
                    "candidate_count_by_family_before_normalization"
                ],
                "candidate_count_by_family_after_normalization": candidate_layer[
                    "candidate_count_by_family_after_normalization"
                ],
                "normalized_candidates_pre_dedup": list(candidate_layer.get("normalized_candidates_pre_dedup", [])),
                "candidates_after_dedup_before_exposure": list(
                    candidate_layer.get("candidates_after_dedup_before_exposure", [])
                ),
                "best_scan_surface_shape_before": best_scan_surface_shape_before,
                "best_verify_surface_shape_before": best_verify_surface_shape_before,
                "best_scan_surface_shape_after": best_scan_surface_shape_after,
                "best_verify_surface_shape_after": best_verify_surface_shape_after,
                "verify_candidate_exposed": verify_candidate_exposed,
                "best_verify_missing_reason": best_verify_missing_reason,
                "verify_lineage_audit": verify_lineage_audit,
                "candidates": candidates,
                "evaluations": evaluations,
                "selector_candidates": selector_candidates,
                "selector_evaluations": selector_evaluations,
                "pre_selector_filter_applied": pre_selector_filter_applied if self.voi_gate_enabled else None,
                "pre_selector_filtered_pairs_empty": (
                    pre_selector_filtered_pairs_empty if self.voi_gate_enabled else None
                ),
                "pre_selector_original_candidate_count": pre_selector_original_candidate_count,
                "pre_selector_filtered_candidate_count": pre_selector_filtered_candidate_count,
                "candidate_backend": candidate_backend,
                "candidate_backend_error": self.llm.last_error,
                "selected_candidate": selection["candidate"],
                "selection_breakdown": selection["score_breakdown"],
                "total_selection_score": selection["total_score"],
                "ask_gate_allowed": ask_gate["allowed"] if self.voi_gate_enabled else None,
                "ask_gate_score": ask_gate["score"] if self.voi_gate_enabled else None,
                "ask_gate_reason": ask_gate["reason"] if self.voi_gate_enabled else None,
                "ask_gate_inputs": ask_gate["inputs"] if self.voi_gate_enabled else None,
                "written_decision_lesson": written_decision_lesson,
                "task_tags": _task_tags(env.task_config, self.mode),
                "info": info,
            }
            step_records.append(step_record)
            observation = next_observation
            last_selection = selection
            last_thin_self_model = thin_self_model_obj.to_dict() if thin_self_model_obj else None
            last_subject_memory = subject_memory
            last_symbolic_other = symbolic_other
            last_lack_snapshot = lack_snapshot
            last_arc_state = dict(arc_state)

        success = total_reward > 0.0 and env.state["done"]
        reflection_backend = "template"
        reflection = build_reflection(
            task=env.task_config,
            mode=self.mode,
            step_records=step_records,
            total_reward=total_reward,
            success=success,
        )
        if self.mode == "no_reflection":
            reflection_backend = "disabled"
        else:
            llm_reflection = self.llm.generate_reflection(
                task=env.task_config,
                mode=self.mode,
                step_records=step_records,
                total_reward=total_reward,
                success=success,
                subject_context={
                    "subject_mode_enabled": self.subject_mode_enabled,
                    "lack_snapshot": last_lack_snapshot,
                    "symbolic_other": last_symbolic_other,
                    "arc_state": last_arc_state,
                }
                if self.subject_mode_enabled
                else None,
            )
            if llm_reflection:
                reflection.update(
                    {
                        "content": str(llm_reflection.get("content", reflection["content"]))[:400],
                        "rule": str(llm_reflection.get("rule", reflection["rule"]))[:240],
                        "avoid_actions": list(llm_reflection.get("avoid_actions", reflection["avoid_actions"]))[:3],
                        "z_delta": dict(llm_reflection.get("z_delta", reflection["z_delta"])),
                    }
                )
                reflection_backend = "openai_responses"
            elif self.llm.configured:
                reflection_backend = "template_fallback"

        admitted = False
        admission_reasons: list[str] = []
        written_entry: dict[str, Any] | None = None
        self_model_delta_applied: dict[str, float] = {}
        commitment_record: dict[str, Any] | None = None
        commitment_written: dict[str, Any] | None = None
        commitment_admitted = False
        commitment_reasons: list[str] = []
        snapshot = self.governance.snapshot(self.memory, self.self_model)

        if self.mode not in {"no_reflection", *VOI_FAMILY_MODES}:
            try:
                admitted, sanitized, admission_reasons = self.governance.admit(
                    reflection,
                    self.memory.read_all(),
                )
                self.governance.audit(
                    "reflection_admission",
                    {
                        "task_id": env.task_id,
                        "mode": self.mode,
                        "admitted": admitted,
                        "reasons": admission_reasons,
                        "content": sanitized["content"],
                    },
                )
                if admitted:
                    written_entry = self.memory.append(sanitized)
                    if not self.subject_mode_enabled and self.mode != "no_self_model":
                        self_model_delta_applied = dict(sanitized.get("z_delta", {}))
                        self.self_model.update(self_model_delta_applied)

                if self.subject_mode_enabled and self.mode != "subject_no_commitment" and last_selection:
                    commitment_record = build_commitment_record(
                        task=env.task_config,
                        selected_candidate=last_selection["candidate"],
                        ranked_candidates=last_selection["ranking"],
                        lack_snapshot=last_lack_snapshot,
                        thin_self_model=last_thin_self_model or {},
                        recognition_judgment=last_selection.get("recognition_judgment", {}),
                    )
                    commitment_entry = build_commitment_entry(
                        task=env.task_config,
                        mode=self.mode,
                        commitment_record=commitment_record,
                    )
                    commitment_admitted, commitment_sanitized, commitment_reasons = self.governance.admit(
                        commitment_entry,
                        self.memory.read_all(),
                    )
                    self.governance.audit(
                        "commitment_admission",
                        {
                            "task_id": env.task_id,
                            "mode": self.mode,
                            "admitted": commitment_admitted,
                            "reasons": commitment_reasons,
                            "content": commitment_sanitized["content"],
                        },
                    )
                    if commitment_admitted:
                        commitment_written = self.memory.append(commitment_sanitized)
            except Exception as exc:  # pragma: no cover
                self.governance.restore(self.memory, self.self_model, snapshot)
                self.governance.audit(
                    "rollback",
                    {
                        "task_id": env.task_id,
                        "mode": self.mode,
                        "error": str(exc),
                    },
                )
                raise

        if self.subject_mode_enabled:
            prior_commitment_count = int(last_arc_state.get("prior_commitment_count", 0))
            if commitment_record:
                identity_continuity_signal = min(
                    1.0,
                    0.30
                    + 0.15 * min(prior_commitment_count, 2)
                    + 0.55 * float(commitment_record.get("bearing_alignment", 0.0)),
                )
            elif last_arc_state.get("episode_index") == 1:
                identity_continuity_signal = 1.0
            else:
                identity_continuity_signal = 0.0
            last_arc_state["identity_continuity_signal"] = round(identity_continuity_signal, 4)

        episode_record = {
            "record_type": "episode",
            "mode": self.mode,
            "task_id": env.task_id,
            "success": success,
            "total_reward": round(total_reward, 4),
            "steps": len(step_records),
            "final_action": step_records[-1]["action"],
            "reflection": reflection,
            "reflection_backend": reflection_backend,
            "reflection_backend_error": self.llm.last_error,
            "reflection_written": written_entry,
            "reflection_admitted": admitted,
            "reflection_reasons": admission_reasons,
            "self_model_before_episode": self_model_before_episode,
            "self_model_after": self.self_model.to_dict(),
            "self_model_delta_applied": self_model_delta_applied,
            "reflection_avoid_actions_written": list(written_entry.get("avoid_actions", [])) if written_entry else [],
            "reflection_z_delta_written": dict(written_entry.get("z_delta", {})) if written_entry else {},
            "retrieved_memory_ids": retrieved_memory_ids,
            "subject_mode_enabled": self.subject_mode_enabled,
            "thin_self_model": last_thin_self_model,
            "subject_memory": last_subject_memory if self.subject_mode_enabled else None,
            "symbolic_other": last_symbolic_other if self.subject_mode_enabled else None,
            "lack_snapshot": last_lack_snapshot if self.subject_mode_enabled else None,
            "commitment_record": commitment_record,
            "commitment_written": commitment_written,
            "commitment_admitted": commitment_admitted,
            "commitment_reasons": commitment_reasons,
            "arc_state": dict(last_arc_state) if self.subject_mode_enabled else None,
        }
        return step_records + [episode_record]
