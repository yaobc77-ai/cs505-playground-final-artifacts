from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class OpenAIResponsesClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5").strip() or "gpt-5"
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "low").strip() or "low"
        self.base_url = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses").strip()
        self.timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
        self.max_retries = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "2")))
        self.retry_backoff_seconds = max(0.0, float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "1.0")))
        self.max_output_tokens_cap = max(256, int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS_CAP", "2400")))
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _post_json(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, bool, str | None]:
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8")), False, None
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            retryable = exc.code in {408, 409, 429, 500, 502, 503, 504} and "insufficient_quota" not in message
            return None, retryable, f"HTTP {exc.code}: {message[:400]}"
        except Exception as exc:  # pragma: no cover - network dependent
            return None, True, str(exc)

    def _extract_output_text(self, body: dict[str, Any]) -> tuple[str, str | None]:
        text_parts: list[str] = []
        for item in body.get("output", []):
            for content in item.get("content", []):
                content_type = content.get("type")
                if content_type == "output_text":
                    text_parts.append(str(content.get("text", "")))
                elif content_type == "refusal":
                    return "", str(content.get("refusal", "model_refusal"))
        raw_text = "".join(text_parts).strip() or str(body.get("output_text", "")).strip()
        return raw_text, None

    def _candidate_json_strings(self, raw_text: str) -> list[str]:
        candidates: list[str] = []

        def add_candidate(value: str) -> None:
            normalized = value.strip().strip("\ufeff")
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        add_candidate(raw_text)

        stripped = raw_text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            inner = stripped[3:-3].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            add_candidate(inner)

        first_object = stripped.find("{")
        last_object = stripped.rfind("}")
        if 0 <= first_object < last_object:
            add_candidate(stripped[first_object : last_object + 1])

        first_array = stripped.find("[")
        last_array = stripped.rfind("]")
        if 0 <= first_array < last_array:
            add_candidate(stripped[first_array : last_array + 1])

        return candidates

    def _parse_json_output(self, raw_text: str) -> tuple[dict[str, Any] | None, json.JSONDecodeError | None]:
        decoder = json.JSONDecoder()
        last_error: json.JSONDecodeError | None = None

        for candidate in self._candidate_json_strings(raw_text):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
            else:
                if isinstance(parsed, dict):
                    return parsed, None

            for opening_char in ("{", "["):
                start = candidate.find(opening_char)
                while start != -1:
                    try:
                        parsed, _ = decoder.raw_decode(candidate[start:])
                    except json.JSONDecodeError as exc:
                        last_error = exc
                        start = candidate.find(opening_char, start + 1)
                        continue
                    if isinstance(parsed, dict):
                        return parsed, None
                    start = candidate.find(opening_char, start + 1)

        return None, last_error

    def _retry_delay(self, attempt: int) -> float:
        return self.retry_backoff_seconds * attempt

    def _bump_output_tokens(self, current_max_output_tokens: int) -> int:
        grown = max(current_max_output_tokens + 200, int(current_max_output_tokens * 1.8))
        return min(grown, self.max_output_tokens_cap)

    def _format_parse_error(
        self,
        *,
        attempt: int,
        total_attempts: int,
        raw_text: str,
        parse_error: json.JSONDecodeError | None,
    ) -> str:
        preview = " ".join(raw_text.split())[:220]
        detail = str(parse_error) if parse_error else "unknown_json_error"
        return f"invalid_json[{attempt}/{total_attempts}]: {detail}; raw={preview}"

    def _request_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any] | None:
        if not self.configured:
            return None

        total_attempts = self.max_retries + 1
        current_max_output_tokens = max_output_tokens
        retry_note = ""

        for attempt in range(1, total_attempts + 1):
            current_instructions = instructions
            if retry_note:
                current_instructions = (
                    f"{instructions}\n"
                    f"{retry_note}"
                )

            payload = {
                "model": self.model,
                "reasoning": {"effort": self.reasoning_effort},
                "instructions": current_instructions,
                "input": input_text,
                "max_output_tokens": current_max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            }

            body, retryable_error, request_error = self._post_json(payload)
            if body is None:
                self.last_error = request_error
                if retryable_error and attempt < total_attempts:
                    time.sleep(self._retry_delay(attempt))
                    continue
                return None

            if body.get("error"):
                error_text = json.dumps(body["error"], ensure_ascii=False)[:400]
                self.last_error = f"response_error: {error_text}"
                if attempt < total_attempts:
                    retry_note = (
                        "The previous attempt returned an internal response error. "
                        "Return one compact JSON object only, with no markdown fences or commentary."
                    )
                    time.sleep(self._retry_delay(attempt))
                    continue
                return None

            if body.get("status") == "incomplete":
                incomplete_reason = str((body.get("incomplete_details") or {}).get("reason", "unknown"))
                self.last_error = f"incomplete_response[{incomplete_reason}]"
                if attempt < total_attempts:
                    retry_note = (
                        "The previous attempt was incomplete. "
                        "Return one compact JSON object only, with no markdown fences or commentary."
                    )
                    if incomplete_reason == "max_output_tokens":
                        current_max_output_tokens = self._bump_output_tokens(current_max_output_tokens)
                    time.sleep(self._retry_delay(attempt))
                    continue
                return None

            raw_text, refusal = self._extract_output_text(body)
            if refusal:
                self.last_error = refusal
                return None

            if not raw_text:
                self.last_error = "empty_model_output"
                if attempt < total_attempts:
                    retry_note = (
                        "The previous attempt returned no usable text. "
                        "Return one compact JSON object only, with no markdown fences or commentary."
                    )
                    time.sleep(self._retry_delay(attempt))
                    continue
                return None

            parsed, parse_error = self._parse_json_output(raw_text)
            if parsed is not None:
                self.last_error = None
                return parsed

            self.last_error = self._format_parse_error(
                attempt=attempt,
                total_attempts=total_attempts,
                raw_text=raw_text,
                parse_error=parse_error,
            )
            if attempt < total_attempts:
                retry_note = (
                    "The previous attempt returned malformed JSON. "
                    "Return one compact JSON object only, with no markdown fences, no prose, and no trailing text."
                )
                current_max_output_tokens = self._bump_output_tokens(current_max_output_tokens)
                time.sleep(self._retry_delay(attempt))
                continue
            return None

        return None

    def generate_candidates(
        self,
        *,
        task: dict[str, Any],
        observation: dict[str, Any],
        self_model: dict[str, float],
        memory_entries: list[dict[str, Any]],
        candidate_limit: int,
        subject_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        available_actions = sorted(observation.get("options", []))
        allowed_actions = sorted(set(available_actions + ["CHOOSE_BEST"]))
        schema = {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": candidate_limit,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "actions": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 2,
                                "items": {"type": "string", "enum": allowed_actions},
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "ask_only",
                                    "choose_now",
                                    "choose_alt",
                                    "ask_then_choose",
                                    "ask_then_choose_fixed",
                                ],
                            },
                        },
                        "required": ["name", "actions", "category"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }

        memory_rules = [entry.get("rule", "") for entry in memory_entries[:3] if entry.get("rule")]
        subject_context = dict(subject_context or {})
        tool_descriptors: list[dict[str, Any]] = []
        clues_by_id = {
            int(clue["id"]): clue
            for clue in task.get("clues", [])
            if clue.get("id") is not None
        }
        for action in available_actions:
            if action.startswith("ASK_SCAN_"):
                clue_id = int(action.removeprefix("ASK_SCAN_"))
                clue = clues_by_id.get(clue_id, {})
                tool_descriptors.append(
                    {
                        "action": action,
                        "tool_family": "scan",
                        "cost": clue.get("scan_cost", task.get("ask_cost", -0.05)),
                        "reliability": clue.get("scan_reliability"),
                    }
                )
            elif action.startswith("ASK_VERIFY_"):
                clue_id = int(action.removeprefix("ASK_VERIFY_"))
                clue = clues_by_id.get(clue_id, {})
                tool_descriptors.append(
                    {
                        "action": action,
                        "tool_family": "verify",
                        "cost": clue.get("verify_cost", task.get("ask_cost", -0.05)),
                        "reliability": clue.get("verify_reliability"),
                    }
                )
            elif action.startswith("ASK_CLUE_"):
                clue_id = int(action.removeprefix("ASK_CLUE_"))
                clue = clues_by_id.get(clue_id, {})
                tool_descriptors.append(
                    {
                        "action": action,
                        "tool_family": "clue",
                        "cost": task.get("ask_cost", -0.05),
                        "reliability": clue.get("reliability"),
                    }
                )
        prompt_lines = [
            f"Task ID: {task['task_id']}",
            f"Difficulty: {task.get('difficulty', 'unknown')}",
            f"Noise level: {task.get('noise_level', 0.0)}",
            f"Ask cost: {task.get('ask_cost', -0.05)}",
            f"Observation:\n{observation['text']}",
            f"Self model: {json.dumps(self_model, ensure_ascii=False)}",
            f"Relevant memory rules: {json.dumps(memory_rules, ensure_ascii=False)}",
            f"Available actions: {json.dumps(available_actions, ensure_ascii=False)}",
        ]
        if tool_descriptors:
            prompt_lines.append(
                f"Tool descriptors: {json.dumps(tool_descriptors, ensure_ascii=False)}"
            )
        if task.get("benchmark_family"):
            prompt_lines.append(f"Benchmark family: {task['benchmark_family']}")
        if task.get("arc_id"):
            prompt_lines.append(f"Arc: {task['arc_id']} / episode {task.get('episode_index')}")
        if subject_context.get("subject_mode_enabled"):
            prompt_lines.append(
                f"Subject memory: {json.dumps(subject_context.get('subject_memory', {}), ensure_ascii=False)}"
            )
            prompt_lines.append(
                f"Symbolic other: {json.dumps(subject_context.get('symbolic_other', {}), ensure_ascii=False)}"
            )
            prompt_lines.append(
                f"Lack snapshot: {json.dumps(subject_context.get('lack_snapshot', {}), ensure_ascii=False)}"
            )
        prompt = "\n".join(
            prompt_lines
            + [
                (
                    "Return 3 to 5 distinct candidate plans. "
                    "Plans may use CHOOSE_BEST only as the second action after an ASK action."
                ),
                (
                    "If ASK_SCAN/ASK_VERIFY tools are available, treat SCAN as cheaper and noisier, "
                    "and VERIFY as more expensive and more reliable. Consider direct CHOOSE, at least one "
                    "SCAN-based plan, and at least one VERIFY-based plan when they are available."
                ),
            ]
        )

        result = self._request_json(
            instructions=(
                "You are the candidate generation module for a controllable agent. "
                "Return only valid JSON that obeys the schema."
            ),
            input_text=prompt,
            schema_name="candidate_bundle",
            schema=schema,
            max_output_tokens=600,
        )
        if not result:
            return None

        cleaned: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for index, candidate in enumerate(result.get("candidates", []), start=1):
            actions = [action for action in candidate.get("actions", []) if action in allowed_actions]
            if not actions:
                continue
            if len(actions) == 2 and actions[0].startswith("CHOOSE"):
                continue
            if "CHOOSE_BEST" in actions[:-1]:
                continue
            key = tuple(actions)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(
                {
                    "name": str(candidate.get("name") or f"llm_candidate_{index}")[:80],
                    "actions": actions[:2],
                    "category": candidate.get("category", "choose_now"),
                }
            )
            if len(cleaned) >= candidate_limit:
                break
        return cleaned or None

    def generate_reflection(
        self,
        *,
        task: dict[str, Any],
        mode: str,
        step_records: list[dict[str, Any]],
        total_reward: float,
        success: bool,
        subject_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        action_enums = sorted({"ASK", "CHOOSE"} | {record["action"] for record in step_records})
        z_delta_keys = ["info_seeking", "cost_sensitivity", "confidence_threshold"]
        schema = {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "rule": {"type": "string"},
                "avoid_actions": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {"type": "string", "enum": action_enums},
                },
                "z_delta": {
                    "type": "object",
                    "properties": {
                        "info_seeking": {"type": "number"},
                        "cost_sensitivity": {"type": "number"},
                        "confidence_threshold": {"type": "number"},
                    },
                    "required": z_delta_keys,
                    "additionalProperties": False,
                },
            },
            "required": ["content", "rule", "avoid_actions", "z_delta"],
            "additionalProperties": False,
        }

        trajectory = [
            {
                "step": record["step"],
                "action": record["action"],
                "reward": record["reward"],
                "belief": record["belief"],
            }
            for record in step_records
        ]
        subject_context = dict(subject_context or {})
        prompt_lines = [
            f"Task ID: {task['task_id']}",
            f"Mode: {mode}",
            f"Success: {success}",
            f"Total reward: {total_reward}",
            f"Trajectory: {json.dumps(trajectory, ensure_ascii=False)}",
        ]
        if task.get("benchmark_family"):
            prompt_lines.append(f"Benchmark family: {task['benchmark_family']}")
        if task.get("arc_id"):
            prompt_lines.append(f"Arc: {task['arc_id']} / episode {task.get('episode_index')}")
        if subject_context.get("subject_mode_enabled"):
            prompt_lines.append(
                f"Subject context: {json.dumps(subject_context, ensure_ascii=False)}"
            )
        prompt = "\n".join(
            prompt_lines
            + [
                (
                    "Write one concise structured reflection. "
                    "If the main lesson is about over-asking, use avoid_actions=['ASK']; "
                    "if it is about choosing too early, use avoid_actions=['CHOOSE']. "
                    "Always include all three z_delta keys and use 0.0 for any key with no change."
                ),
            ]
        )

        result = self._request_json(
            instructions=(
                "You are the reflection-writing module for a controllable agent. "
                "Return only valid JSON that obeys the schema."
            ),
            input_text=prompt,
            schema_name="reflection_bundle",
            schema=schema,
            max_output_tokens=400,
        )
        if not result:
            return None

        raw_z_delta = result.get("z_delta", {})
        cleaned_z_delta: dict[str, float] = {}
        for key in z_delta_keys:
            value = raw_z_delta.get(key, 0.0)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cleaned_z_delta[key] = float(value)
            else:
                cleaned_z_delta[key] = 0.0

        return {
            "content": str(result.get("content", ""))[:400],
            "rule": str(result.get("rule", ""))[:240],
            "avoid_actions": [
                action
                for action in result.get("avoid_actions", [])
                if isinstance(action, str) and action in action_enums
            ][:3],
            "z_delta": cleaned_z_delta,
        }
