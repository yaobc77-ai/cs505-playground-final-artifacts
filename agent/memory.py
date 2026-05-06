from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.voi_memory import (
    CAUTION_LESSON_TYPES,
    POSITIVE_LESSON_TYPES,
    decision_lesson_key,
    decision_lesson_match_score,
    retention_score,
    summarize_decision_lesson_signal,
)


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def read_all(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    def _write_all(self, entries: list[dict[str, Any]]) -> None:
        ordered = sorted(entries, key=lambda entry: int(entry.get("version", 0)))
        with self.path.open("w", encoding="utf-8") as handle:
            for entry in ordered:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def latest_version(self, entries: list[dict[str, Any]] | None = None) -> int:
        entries = entries if entries is not None else self.read_all()
        return max((int(entry.get("version", 0)) for entry in entries), default=0)

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = dict(entry)
        version = self.latest_version() + 1
        entry.setdefault("memory_id", f"mem-{version:05d}")
        entry.setdefault("version", version)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def search(
        self,
        query_terms: list[str] | None = None,
        task_tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        entries = self.read_all()
        query_terms = [term.lower() for term in (query_terms or []) if term]
        task_tags = [tag.lower() for tag in (task_tags or []) if tag]
        scored: list[tuple[float, dict[str, Any]]] = []

        for entry in entries:
            if entry.get("type") == "decision_lesson":
                continue
            score = 0.0
            content = str(entry.get("content", "")).lower()
            entry_tags = [str(tag).lower() for tag in entry.get("task_tags", [])]

            for term in query_terms:
                if term in content:
                    score += 2.0
                if term in entry_tags:
                    score += 1.0

            for tag in task_tags:
                if tag in entry_tags:
                    score += 1.5

            if entry.get("type") == "reflection":
                score += 0.1

            if score > 0 or (not query_terms and not task_tags):
                scored.append((score, entry))

        scored.sort(
            key=lambda item: (item[0], item[1].get("version", 0)),
            reverse=True,
        )
        return [entry for _, entry in scored[:limit]]

    def _prune_decision_lessons(
        self,
        entries: list[dict[str, Any]],
        *,
        current_version: int,
    ) -> list[dict[str, Any]]:
        keep_ids: set[str] = set()
        for lesson_type in set(POSITIVE_LESSON_TYPES + CAUTION_LESSON_TYPES):
            typed_entries = [
                entry
                for entry in entries
                if entry.get("type") == "decision_lesson"
                and str(entry.get("lesson_type")) == lesson_type
            ]
            typed_entries.sort(
                key=lambda entry: (
                    retention_score(entry, current_version=current_version),
                    int(entry.get("last_seen_version", entry.get("version", 0)) or 0),
                    int(entry.get("version", 0)),
                ),
                reverse=True,
            )
            keep_ids.update(
                str(entry.get("memory_id"))
                for entry in typed_entries[:40]
                if entry.get("memory_id")
            )

        pruned: list[dict[str, Any]] = []
        for entry in entries:
            if entry.get("type") != "decision_lesson":
                pruned.append(entry)
                continue
            if str(entry.get("memory_id")) in keep_ids:
                pruned.append(entry)
        return pruned

    def upsert_decision_lesson(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        if entry.get("type") != "decision_lesson":
            raise ValueError("upsert_decision_lesson expects type='decision_lesson'")

        entries = self.read_all()
        current_version = self.latest_version(entries) + 1
        incoming = dict(entry)
        incoming["strength"] = max(0.0, min(1.0, float(incoming.get("strength", 0.0))))

        match_index: int | None = None
        match_key = decision_lesson_key(incoming)
        for index, existing in enumerate(entries):
            if existing.get("type") != "decision_lesson":
                continue
            if decision_lesson_key(existing) == match_key:
                match_index = index
                break

        updated: dict[str, Any]
        if match_index is not None:
            existing = dict(entries[match_index])
            old_strength = float(existing.get("strength", 0.0))
            new_strength = float(incoming.get("strength", 0.0))
            updated = {
                **existing,
                **incoming,
                "memory_id": existing.get("memory_id", incoming.get("memory_id", f"mem-{current_version:05d}")),
                "version": current_version,
                "last_seen_version": current_version,
                "reinforcement_count": int(existing.get("reinforcement_count", 0)) + 1,
                "strength": min(1.0, 0.7 * old_strength + 0.3 * new_strength + 0.05),
            }
            entries[match_index] = updated
        else:
            updated = {
                **incoming,
                "memory_id": incoming.get("memory_id", f"mem-{current_version:05d}"),
                "version": current_version,
                "last_seen_version": current_version,
                "reinforcement_count": int(incoming.get("reinforcement_count", 1)),
            }
            entries.append(updated)

        entries = self._prune_decision_lessons(entries, current_version=current_version)
        self._write_all(entries)
        updated_id = str(updated.get("memory_id", ""))
        for saved_entry in entries:
            if str(saved_entry.get("memory_id", "")) == updated_id:
                return saved_entry
        return None

    def search_decision_lessons(
        self,
        *,
        query_scope: dict[str, Any],
        positive_limit: int = 2,
        caution_limit: int = 2,
    ) -> dict[str, Any]:
        entries = self.read_all()
        current_version = self.latest_version(entries)
        normalized_scope = dict(query_scope)
        normalized_scope.setdefault("tool_family", "any")
        normalized_scope.setdefault("reliability_bucket", "")
        normalized_scope.setdefault("conflict_bucket", "")
        scored_positive: list[tuple[float, dict[str, Any]]] = []
        scored_caution: list[tuple[float, dict[str, Any]]] = []

        for entry in entries:
            if entry.get("type") != "decision_lesson":
                continue
            score = decision_lesson_match_score(
                entry,
                query_scope=normalized_scope,
                current_version=current_version,
            )
            bucket = (
                scored_positive
                if str(entry.get("lesson_type")) in POSITIVE_LESSON_TYPES
                else scored_caution
            )
            bucket.append((score, entry))

        sort_key = lambda item: (
            item[0],
            int(item[1].get("last_seen_version", item[1].get("version", 0)) or 0),
            int(item[1].get("version", 0)),
        )
        scored_positive.sort(key=sort_key, reverse=True)
        scored_caution.sort(key=sort_key, reverse=True)
        positive = [entry for _, entry in scored_positive[:positive_limit]]
        caution = [entry for _, entry in scored_caution[:caution_limit]]
        summary = summarize_decision_lesson_signal(
            positive_lessons=positive,
            caution_lessons=caution,
        )
        return {
            "query_scope": normalized_scope,
            "query_tool_family": str(normalized_scope.get("tool_family", "any")),
            "positive": positive,
            "caution": caution,
            "summary": summary,
            "positive_tool_families": sorted(
                {
                    str(entry.get("tool_family", ""))
                    for entry in positive
                    if entry.get("tool_family")
                }
            ),
            "caution_tool_families": sorted(
                {
                    str(entry.get("tool_family", ""))
                    for entry in caution
                    if entry.get("tool_family")
                }
            ),
        }
