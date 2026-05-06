from __future__ import annotations

from typing import Any


TOOL_FAMILIES = ("scan", "verify", "clue", "choose", "other")
VERIFY_PRUNED_STAGES = (
    "none",
    "generated",
    "family_merge",
    "normalization_merge",
    "dedup",
    "candidate_cap",
    "pre_selector_filter",
    "unknown",
)
VERIFY_PRUNED_REASONS = (
    "none",
    "shape_replaced_by_equivalent",
    "dominated_within_family",
    "dominated_cross_family",
    "deduplicated",
    "budget_cap",
    "filter_rule",
    "unknown",
)


def action_tool_family(action: str | None) -> str:
    action = str(action or "")
    if action.startswith("ASK_SCAN_"):
        return "scan"
    if action.startswith("ASK_VERIFY_"):
        return "verify"
    if action.startswith("ASK_CLUE_"):
        return "clue"
    if action.startswith("CHOOSE_"):
        return "choose"
    return "other"


def verify_lineage_key(action: str | None) -> str | None:
    action = str(action or "")
    if action.startswith("ASK_VERIFY_"):
        return action
    return None


def candidate_tool_family(candidate: dict[str, Any] | None) -> str:
    actions = [str(action) for action in (candidate or {}).get("actions", [])]
    if not actions:
        return "other"
    return action_tool_family(actions[0])


def candidate_verify_lineage_key(candidate: dict[str, Any] | None) -> str | None:
    actions = [str(action) for action in (candidate or {}).get("actions", [])]
    if not actions:
        return None
    return verify_lineage_key(actions[0])


def candidate_surface_shape(candidate: dict[str, Any] | None) -> str:
    if not candidate:
        return "none"
    actions = [str(action) for action in candidate.get("actions", [])]
    category = str(candidate.get("category", ""))
    if not actions:
        if category == "ask_only":
            return "ask_only"
        if category == "ask_then_choose_fixed":
            return "fixed_choose"
        if category == "ask_then_choose":
            return "canonical_choose_best"
        return "none"
    family = action_tool_family(actions[0])
    if family not in {"scan", "verify", "clue"}:
        return "none"
    if len(actions) == 1:
        return "ask_only"
    if len(actions) == 2 and actions[1] == "CHOOSE_BEST":
        return "canonical_choose_best"
    if len(actions) == 2 and actions[1].startswith("CHOOSE_") and actions[1] != "CHOOSE_BEST":
        return "fixed_choose"
    return "multi_step"


def count_candidates_by_family(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {family: 0 for family in TOOL_FAMILIES}
    for candidate in candidates:
        counts[candidate_tool_family(candidate)] += 1
    return counts


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in candidates:
        actions = tuple(str(action) for action in candidate.get("actions", []))
        if not actions or actions in seen:
            continue
        seen.add(actions)
        deduped.append(dict(candidate))
    return deduped


def normalize_candidate_shape(candidate: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized = dict(candidate)
    actions = [str(action) for action in candidate.get("actions", [])]
    category = str(candidate.get("category", ""))
    normalized["_coverage_original_actions"] = list(actions)
    normalized["_coverage_original_category"] = category
    normalized["_coverage_original_shape"] = candidate_surface_shape(candidate)
    family = action_tool_family(actions[0]) if actions else "other"
    if family not in {"scan", "verify", "clue"} or not actions:
        normalized["_coverage_normalized"] = False
        return normalized, False
    if len(actions) == 2 and actions[1] == "CHOOSE_BEST":
        normalized["_coverage_normalized"] = False
        return normalized, False
    normalized["actions"] = [actions[0], "CHOOSE_BEST"]
    normalized["category"] = "ask_then_choose"
    normalized["name"] = str(candidate.get("name") or f"{actions[0].lower()}_candidate")[:60] + "_canonical"
    normalized["_coverage_normalized"] = True
    normalized["_coverage_normalized_from_shape"] = candidate_surface_shape(candidate)
    return normalized, True


def normalize_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_pre_dedup: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for candidate in candidates:
        updated, changed = normalize_candidate_shape(candidate)
        normalized_pre_dedup.append(updated)
        if changed:
            traces.append(
                {
                    "name": str(candidate.get("name", "")),
                    "from_actions": list(candidate.get("actions", [])),
                    "to_actions": list(updated.get("actions", [])),
                    "from_category": str(candidate.get("category", "")),
                    "to_category": str(updated.get("category", "")),
                    "from_shape": candidate_surface_shape(candidate),
                    "to_shape": candidate_surface_shape(updated),
                    "tool_family": candidate_tool_family(updated),
                }
            )
    return dedupe_candidates(normalized_pre_dedup), traces


def _normalize_candidates_with_pre_dedup(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_pre_dedup: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for candidate in candidates:
        updated, changed = normalize_candidate_shape(candidate)
        normalized_pre_dedup.append(updated)
        if changed:
            traces.append(
                {
                    "name": str(candidate.get("name", "")),
                    "from_actions": list(candidate.get("actions", [])),
                    "to_actions": list(updated.get("actions", [])),
                    "from_category": str(candidate.get("category", "")),
                    "to_category": str(updated.get("category", "")),
                    "from_shape": candidate_surface_shape(candidate),
                    "to_shape": candidate_surface_shape(updated),
                    "tool_family": candidate_tool_family(updated),
                }
            )
    return normalized_pre_dedup, dedupe_candidates(normalized_pre_dedup), traces


def build_canonical_exposure_candidate(
    *,
    action: str,
    family: str,
    source: str,
) -> dict[str, Any]:
    return {
        "name": f"{action.lower()}_canonical_exposed",
        "actions": [action, "CHOOSE_BEST"],
        "category": "ask_then_choose",
        "_coverage_injected": True,
        "_coverage_injected_family": family,
        "_coverage_injected_source": source,
        "_coverage_original_actions": [action, "CHOOSE_BEST"],
        "_coverage_original_category": "ask_then_choose",
        "_coverage_original_shape": "canonical_choose_best",
        "_coverage_normalized": False,
    }


def ensure_family_exposure(
    *,
    candidates: list[dict[str, Any]],
    available_actions: list[str],
    template_candidates: list[dict[str, Any]] | None = None,
    families: tuple[str, ...] = ("scan", "verify"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_candidates = list(dedupe_candidates(candidates))
    template_candidates = list(template_candidates or [])
    available_by_family: dict[str, list[str]] = {family: [] for family in families}
    for action in available_actions:
        family = action_tool_family(action)
        if family in available_by_family:
            available_by_family[family].append(str(action))

    injected: list[dict[str, Any]] = []
    injected_families: list[str] = []
    for family in families:
        if not available_by_family.get(family):
            continue
        family_present = any(candidate_tool_family(candidate) == family for candidate in final_candidates)
        if family_present:
            continue
        template_match = next(
            (
                candidate
                for candidate in template_candidates
                if candidate_tool_family(candidate) == family
            ),
            None,
        )
        if template_match is not None:
            injected_candidate, _ = normalize_candidate_shape(template_match)
            injected_candidate["_coverage_injected"] = True
            injected_candidate["_coverage_injected_family"] = family
            injected_candidate["_coverage_injected_source"] = "template"
        else:
            injected_candidate = build_canonical_exposure_candidate(
                action=available_by_family[family][0],
                family=family,
                source="observation",
            )
        final_candidates.append(injected_candidate)
        injected.append(injected_candidate)
        injected_families.append(family)

    return dedupe_candidates(final_candidates), {
        "injected_candidates": injected,
        "injected_families": injected_families,
    }


def build_candidate_layer_bundle(
    *,
    base_candidates: list[dict[str, Any]],
    available_actions: list[str],
    template_candidates: list[dict[str, Any]] | None = None,
    families: tuple[str, ...] = ("scan", "verify"),
) -> dict[str, Any]:
    source_candidates = dedupe_candidates(base_candidates)
    counts_before = count_candidates_by_family(source_candidates)
    normalized_pre_dedup, normalized_after_dedup, normalization_traces = _normalize_candidates_with_pre_dedup(
        source_candidates
    )
    after_candidates, exposure_info = ensure_family_exposure(
        candidates=normalized_after_dedup,
        available_actions=available_actions,
        template_candidates=template_candidates,
        families=families,
    )
    counts_after = count_candidates_by_family(after_candidates)
    return {
        "source_candidates": source_candidates,
        "candidates_before_normalization": source_candidates,
        "candidates_after_family_merge": source_candidates,
        "normalized_candidates_pre_dedup": normalized_pre_dedup,
        "candidates_after_dedup_before_exposure": normalized_after_dedup,
        "candidates_after_normalization": after_candidates,
        "candidate_count_by_family_before_normalization": counts_before,
        "candidate_count_by_family_after_normalization": counts_after,
        "candidate_normalization_trace": normalization_traces,
        "candidate_exposure_trace": exposure_info,
    }


def best_family_surface_shape(
    candidates: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    family: str,
) -> str:
    best_score: float | None = None
    best_shape = "none"
    for candidate, evaluation in zip(candidates, evaluations):
        if candidate_tool_family(candidate) != family:
            continue
        score = float(evaluation.get("score", 0.0))
        if best_score is None or score > best_score:
            best_score = score
            best_shape = candidate_surface_shape(candidate)
    return best_shape


def infer_verify_candidate_exposed(selector_candidates: list[dict[str, Any]]) -> bool:
    return any(candidate_tool_family(candidate) == "verify" for candidate in selector_candidates)


def infer_best_verify_missing_reason(
    *,
    remaining_verify_actions: list[str],
    candidates_before_normalization: list[dict[str, Any]],
    candidates_after_normalization: list[dict[str, Any]],
    selector_candidates: list[dict[str, Any]],
    selector_evaluations: list[dict[str, Any]],
) -> str:
    if not remaining_verify_actions:
        return "not_applicable"

    before_verify = [
        candidate
        for candidate in candidates_before_normalization
        if candidate_tool_family(candidate) == "verify"
    ]
    after_verify = [
        candidate
        for candidate in candidates_after_normalization
        if candidate_tool_family(candidate) == "verify"
    ]
    selector_verify = [
        candidate
        for candidate in selector_candidates
        if candidate_tool_family(candidate) == "verify"
    ]
    before_has_canonical = any(
        candidate_surface_shape(candidate) == "canonical_choose_best"
        for candidate in before_verify
    )

    if not before_verify:
        return "not_generated"
    if not before_has_canonical and not after_verify:
        return "generated_noncanonical"
    if after_verify and not selector_verify:
        return "generated_but_pruned"
    if selector_verify:
        evaluation_by_name = {
            str(evaluation.get("candidate_name", "")): evaluation
            for evaluation in selector_evaluations
        }
        best_verify_score: float | None = None
        best_other_score: float | None = None
        for candidate in selector_candidates:
            name = str(candidate.get("name", ""))
            score = float(evaluation_by_name.get(name, {}).get("score", 0.0))
            if candidate_tool_family(candidate) == "verify":
                if best_verify_score is None or score > best_verify_score:
                    best_verify_score = score
            else:
                if best_other_score is None or score > best_other_score:
                    best_other_score = score
        if (
            best_verify_score is not None
            and best_other_score is not None
            and best_other_score > best_verify_score
        ):
            return "dominated_after_scoring"
        return "none"
    if not before_has_canonical:
        return "generated_noncanonical"
    return "generated_but_pruned"


def _lineage_present(candidates: list[dict[str, Any]], lineage_key: str) -> bool:
    return any(candidate_verify_lineage_key(candidate) == lineage_key for candidate in candidates)


def _lineage_shapes(candidates: list[dict[str, Any]], lineage_key: str) -> list[str]:
    shapes = [
        candidate_surface_shape(candidate)
        for candidate in candidates
        if candidate_verify_lineage_key(candidate) == lineage_key
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for shape in shapes:
        if shape in seen:
            continue
        seen.add(shape)
        ordered.append(shape)
    return ordered


def _lineage_actions(candidates: list[dict[str, Any]], lineage_key: str) -> list[list[str]]:
    actions_list = [
        [str(action) for action in candidate.get("actions", [])]
        for candidate in candidates
        if candidate_verify_lineage_key(candidate) == lineage_key
    ]
    seen: set[tuple[str, ...]] = set()
    ordered: list[list[str]] = []
    for actions in actions_list:
        key = tuple(actions)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(actions)
    return ordered


def _resolve_active_generation_sources(
    *,
    candidate_backend: str,
    template_candidates: list[dict[str, Any]],
    llm_candidates: list[dict[str, Any]],
    source_candidates: list[dict[str, Any]],
    final_candidates: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    template_candidates = dedupe_candidates(list(template_candidates))
    llm_candidates = dedupe_candidates(list(llm_candidates))
    source_candidates = dedupe_candidates(list(source_candidates))
    final_candidates = dedupe_candidates(list(final_candidates))

    if candidate_backend == "openai_responses_plus_template":
        merged = source_candidates or dedupe_candidates(llm_candidates + template_candidates)
        return {"llm": llm_candidates, "template": template_candidates}, merged
    if candidate_backend == "openai_responses":
        active = llm_candidates or source_candidates or final_candidates
        return {"llm": active}, active
    active = template_candidates or source_candidates or final_candidates
    return {"template": active}, active


def _infer_candidate_cap_reason(active_source_candidates: list[dict[str, Any]]) -> str:
    verify_count = sum(
        1
        for candidate in active_source_candidates
        if candidate_tool_family(candidate) == "verify"
    )
    other_count = sum(
        1
        for candidate in active_source_candidates
        if candidate_tool_family(candidate) != "verify"
    )
    if verify_count > 0 and other_count == 0:
        return "dominated_within_family"
    if verify_count == 0 and other_count > 0:
        return "dominated_cross_family"
    return "budget_cap"


def build_verify_lineage_survival(
    *,
    remaining_verify_actions: list[str],
    candidate_backend: str,
    candidate_limit: int,
    template_candidates: list[dict[str, Any]] | None = None,
    llm_candidates: list[dict[str, Any]] | None = None,
    source_candidates: list[dict[str, Any]] | None = None,
    normalized_candidates_pre_dedup: list[dict[str, Any]] | None = None,
    deduped_candidates: list[dict[str, Any]] | None = None,
    final_candidates: list[dict[str, Any]] | None = None,
    selector_candidates: list[dict[str, Any]] | None = None,
    ask_gate_allowed: bool | None = None,
    pre_selector_filter_evidence: bool | None = None,
) -> list[dict[str, Any]]:
    seen_lineages: set[str] = set()
    lineage_keys: list[str] = []
    for action in remaining_verify_actions:
        lineage_key = verify_lineage_key(action)
        if not lineage_key or lineage_key in seen_lineages:
            continue
        seen_lineages.add(lineage_key)
        lineage_keys.append(lineage_key)

    template_candidates = dedupe_candidates(list(template_candidates) if template_candidates is not None else [])
    llm_candidates = dedupe_candidates(list(llm_candidates) if llm_candidates is not None else [])
    source_candidates = dedupe_candidates(list(source_candidates) if source_candidates is not None else [])
    final_candidates = dedupe_candidates(list(final_candidates) if final_candidates is not None else [])
    active_sources, family_merge_candidates = _resolve_active_generation_sources(
        candidate_backend=candidate_backend,
        template_candidates=template_candidates,
        llm_candidates=llm_candidates,
        source_candidates=source_candidates,
        final_candidates=final_candidates,
    )
    normalized_candidates_pre_dedup = (
        list(normalized_candidates_pre_dedup)
        if normalized_candidates_pre_dedup is not None
        else list(family_merge_candidates)
    )
    deduped_candidates = dedupe_candidates(
        list(deduped_candidates)
        if deduped_candidates is not None
        else list(normalized_candidates_pre_dedup)
    )
    selector_candidates = dedupe_candidates(
        list(selector_candidates)
        if selector_candidates is not None
        else list(final_candidates)
    )

    nonempty_active_sources = [
        dedupe_candidates(list(candidates))
        for candidates in active_sources.values()
        if candidates
    ]
    active_source_candidates = dedupe_candidates(
        [candidate for candidates in nonempty_active_sources for candidate in candidates]
    )
    candidate_limit = max(int(candidate_limit or 0), 0)
    candidate_cap_inferred = bool(
        candidate_limit
        and nonempty_active_sources
        and all(len(candidates) >= candidate_limit for candidates in nonempty_active_sources)
    )

    rows: list[dict[str, Any]] = []
    for lineage_key in lineage_keys:
        generation_sources = [
            source_name
            for source_name, candidates in active_sources.items()
            if _lineage_present(dedupe_candidates(list(candidates)), lineage_key)
        ]
        verify_generated = bool(generation_sources)
        verify_retained_after_family_merge = _lineage_present(family_merge_candidates, lineage_key)
        verify_normalized = _lineage_present(normalized_candidates_pre_dedup, lineage_key)
        verify_retained_after_dedup = _lineage_present(deduped_candidates, lineage_key)
        verify_retained_before_selector = _lineage_present(final_candidates, lineage_key)
        verify_entered_selector = _lineage_present(selector_candidates, lineage_key)
        source_shapes = _lineage_shapes(family_merge_candidates, lineage_key)
        normalized_shapes = _lineage_shapes(normalized_candidates_pre_dedup, lineage_key)
        selector_shapes = _lineage_shapes(selector_candidates, lineage_key)
        shape_replacement_detected = bool(
            any(shape != "canonical_choose_best" for shape in source_shapes)
            and "canonical_choose_best" in normalized_shapes
        )
        verify_injected_before_selector = bool(
            not verify_retained_after_dedup and verify_retained_before_selector
        )

        verify_pruned_stage = "none"
        verify_pruned_reason = "none"
        if not verify_entered_selector:
            if verify_retained_before_selector:
                verify_pruned_stage = "pre_selector_filter"
                verify_pruned_reason = (
                    "filter_rule"
                    if bool(pre_selector_filter_evidence)
                    else "unknown"
                )
            elif not verify_retained_after_dedup:
                if not verify_normalized:
                    if not verify_retained_after_family_merge:
                        if not verify_generated:
                            if candidate_cap_inferred:
                                verify_pruned_stage = "candidate_cap"
                                verify_pruned_reason = _infer_candidate_cap_reason(active_source_candidates)
                            else:
                                verify_pruned_stage = "generated"
                                verify_pruned_reason = "unknown"
                        else:
                            verify_pruned_stage = "family_merge"
                            verify_pruned_reason = "unknown"
                    else:
                        verify_pruned_stage = "normalization_merge"
                        verify_pruned_reason = (
                            "shape_replaced_by_equivalent" if shape_replacement_detected else "unknown"
                        )
                else:
                    verify_pruned_stage = "dedup"
                    verify_pruned_reason = "deduplicated"
            else:
                verify_pruned_stage = "unknown"
                verify_pruned_reason = "unknown"

        rows.append(
            {
                "verify_lineage_key": lineage_key,
                "verify_generated": verify_generated,
                "verify_normalized": verify_normalized,
                "verify_retained_after_family_merge": verify_retained_after_family_merge,
                "verify_retained_after_candidate_cap": verify_generated,
                "verify_retained_after_dedup": verify_retained_after_dedup,
                "verify_retained_before_selector": verify_retained_before_selector,
                "verify_entered_selector": verify_entered_selector,
                "verify_pruned_stage": verify_pruned_stage,
                "verify_pruned_reason": verify_pruned_reason,
                "verify_generation_sources": generation_sources,
                "verify_source_surface_shapes": source_shapes,
                "verify_normalized_surface_shapes": normalized_shapes,
                "verify_selector_surface_shapes": selector_shapes,
                "verify_shape_replacement_detected": shape_replacement_detected,
                "verify_injected_before_selector": verify_injected_before_selector,
                "verify_family_merge_actions": _lineage_actions(family_merge_candidates, lineage_key),
                "verify_selector_actions": _lineage_actions(selector_candidates, lineage_key),
            }
        )

    return rows
