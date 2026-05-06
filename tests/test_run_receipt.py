from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from eval.run_receipt import collect_run_receipt, render_receipt_markdown


SCRATCH_ROOT = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime")


@contextlib.contextmanager
def _scratch_dir(prefix: str):
    path = SCRATCH_ROOT / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_tasks(path: Path, task_ids: list[str]) -> None:
    payload = "\n".join(json.dumps({"task_id": task_id}) for task_id in task_ids) + "\n"
    path.write_text(payload, encoding="utf-8")


def _write_mode_records(path: Path, *, mode: str, tasks_path: Path, task_ids: list[str], truncated: bool = False) -> None:
    lines = [
        json.dumps(
            {
                "record_type": "run_meta",
                "mode": mode,
                "seed": 7,
                "timestamp": "2026-04-15T20:52:32.641462+00:00",
                "tasks_path": str(tasks_path),
            }
        )
    ]
    for task_id in task_ids:
        lines.append(
            json.dumps(
                {
                    "record_type": "episode",
                    "mode": mode,
                    "task_id": task_id,
                    "success": True,
                    "total_reward": 1.0,
                    "steps": 1,
                }
            )
        )
    if truncated:
        lines.append("{bad json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RunReceiptTests(unittest.TestCase):
    def test_intact_jsonl_passes(self) -> None:
        with _scratch_dir("run_receipt_ok") as root:
            tasks_path = root / "tasks.jsonl"
            _write_tasks(tasks_path, ["task-0001", "task-0002"])
            _write_mode_records(root / "full.jsonl", mode="full", tasks_path=tasks_path, task_ids=["task-0001", "task-0002"])
            (root / "pruning_path_audit.md").write_text("# audit\n", encoding="utf-8")
            (root / "verify_survival_rows.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "pre_selector_pruning_audit.md").write_text("# pre-selector audit\n", encoding="utf-8")
            (root / "pre_selector_pruning_rows.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "blocked_step_audit.md").write_text("# blocked step audit\n", encoding="utf-8")
            (root / "blocked_step_audit_rows.jsonl").write_text("{}\n", encoding="utf-8")
            receipt = collect_run_receipt(run_root=root)
            self.assertEqual("pass", receipt["jsonl_integrity"])
            self.assertEqual("pass", receipt["task_id_pairing"])
            self.assertTrue(receipt["safe_for_interpretation"])
            self.assertEqual(2, receipt["expected_episode_count"])
            self.assertEqual(2, receipt["valid_episode_counts"]["0:full"])
            self.assertTrue(receipt["hash_complete"])
            self.assertIn("tasks_path", receipt["artifact_hashes"])
            self.assertIn("pruning_path_audit.md", receipt["reports_generated"])
            self.assertIn("verify_survival_rows.jsonl", receipt["reports_generated"])
            self.assertIn("pre_selector_pruning_audit.md", receipt["reports_generated"])
            self.assertIn("pre_selector_pruning_rows.jsonl", receipt["reports_generated"])
            self.assertIn("blocked_step_audit.md", receipt["reports_generated"])
            self.assertIn("blocked_step_audit_rows.jsonl", receipt["reports_generated"])
            self.assertIn("- jsonl_integrity: pass", render_receipt_markdown(receipt))

    def test_truncated_jsonl_fails_integrity(self) -> None:
        with _scratch_dir("run_receipt_truncated") as root:
            tasks_path = root / "tasks.jsonl"
            _write_tasks(tasks_path, ["task-0001"])
            _write_mode_records(root / "full.jsonl", mode="full", tasks_path=tasks_path, task_ids=["task-0001"], truncated=True)
            receipt = collect_run_receipt(run_root=root)
            self.assertEqual("fail", receipt["jsonl_integrity"])
            self.assertFalse(receipt["safe_for_interpretation"])

    def test_mismatched_task_ids_fail_pairing(self) -> None:
        with _scratch_dir("run_receipt_pairing") as root:
            tasks_path = root / "tasks.jsonl"
            _write_tasks(tasks_path, ["task-0001", "task-0002"])
            _write_mode_records(root / "full.jsonl", mode="full", tasks_path=tasks_path, task_ids=["task-0001", "task-0002"])
            _write_mode_records(root / "selector_penalty.jsonl", mode="selector_penalty", tasks_path=tasks_path, task_ids=["task-0001"])
            receipt = collect_run_receipt(run_root=root)
            self.assertEqual("pass", receipt["jsonl_integrity"])
            self.assertEqual("fail", receipt["task_id_pairing"])
            self.assertIn("task-0002", receipt["missing_tasks"])

    def test_shell_timeout_acceptance_requires_integrity_pairing_and_hashes(self) -> None:
        with _scratch_dir("run_receipt_timeout") as root:
            tasks_path = root / "tasks.jsonl"
            _write_tasks(tasks_path, ["task-0001"])
            _write_mode_records(root / "full.jsonl", mode="full", tasks_path=tasks_path, task_ids=["task-0001"])
            receipt = collect_run_receipt(run_root=root, shell_timeout_observed=True)
            self.assertTrue(receipt["shell_timeout_observed"])
            self.assertTrue(receipt["shell_timeout_accepted_by_integrity_rule"])
            markdown = render_receipt_markdown(receipt)
            self.assertIn("- shell_timeout_accepted_by_integrity_rule: true", markdown)


if __name__ == "__main__":
    unittest.main()
