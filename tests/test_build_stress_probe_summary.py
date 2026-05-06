from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from eval.build_stress_probe_summary import build_stress_probe_summary


SCRATCH_ROOT = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime")


@contextlib.contextmanager
def _scratch_dir(prefix: str):
    path = SCRATCH_ROOT / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_receipt(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Post-Run Receipt",
                "",
                "- run_name: synthetic_stress",
                "- run_completed: true",
                "- jsonl_integrity: pass",
                "- task_id_pairing: pass",
                "- safe_for_interpretation: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _mode_payload(
    *,
    tool_choice: float,
    oracle_choice: float,
    choose_swallow: float,
    family_mismatch: float,
    verify_usage: float,
) -> dict[str, object]:
    return {
        "overall": {
            "tool_choice_step_count": tool_choice,
            "oracle_tool_choice_step_count": oracle_choice,
            "choose_swallow_rate_given_oracle_tool_choice": choose_swallow,
            "selected_vs_oracle_family_mismatch_rate": family_mismatch,
            "verify_usage_rate": verify_usage,
            "wrong_tool_rate": 0.0,
            "tool_family_mismatch_rate_vs_oracle": 0.0,
        },
        "episodes": {"success_rate": 1.0},
    }


def _write_diagnosis(path: Path, *, full: dict[str, object], selector: dict[str, object], voi: dict[str, object], blocked_step_count: int = 10) -> None:
    payload = {
        "modes": {
            "full": full,
            "selector_penalty": selector,
            "voi_memory_v1": voi,
        },
        "blocked_step_audit": {
            "voi_memory_v1": {
                "task_pressure_count": blocked_step_count,
                "surfaced_ask_deficit_count": 0,
                "hard_cutoff_candidate_count": 0,
                "blocked_step_count": blocked_step_count,
                "route_target": "task_pressure",
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BuildStressProbeSummaryTests(unittest.TestCase):
    def test_reopened_separation_routes_to_reference_confirmation(self) -> None:
        with _scratch_dir("stress_probe_route_b") as root:
            _write_receipt(root / "POST_RUN_RECEIPT.md")
            (root / "blocked_step_audit.md").write_text("# blocked\n", encoding="utf-8")
            _write_diagnosis(
                root / "diagnosis.json",
                full=_mode_payload(tool_choice=57, oracle_choice=156, choose_swallow=0.6346, family_mismatch=0.8397, verify_usage=0.3057),
                selector=_mode_payload(tool_choice=29, oracle_choice=129, choose_swallow=0.7752, family_mismatch=0.9225, verify_usage=0.1938),
                voi=_mode_payload(tool_choice=83, oracle_choice=183, choose_swallow=0.5464, family_mismatch=0.5464, verify_usage=0.0984),
                blocked_step_count=100,
            )

            summary = build_stress_probe_summary(run_root=root, delta=0.10)
            self.assertTrue(summary["receipt_ok"] if "receipt_ok" in summary else summary["stress_pack"]["receipt_ok"])
            self.assertTrue(summary["pure_task_pressure"])
            self.assertTrue(summary["separation_reopened"])
            self.assertEqual("B", summary["route"])
            self.assertEqual("prepare_seed11_reference_confirmation_review", summary["next_action"])
            metrics = {item["metric"] for item in summary["trigger_good"]}
            self.assertEqual({"selected_vs_oracle_family_mismatch_rate"}, metrics)
            self.assertEqual([], summary["negative_inversion"])

    def test_negative_inversion_routes_to_c(self) -> None:
        with _scratch_dir("stress_probe_route_c") as root:
            _write_receipt(root / "POST_RUN_RECEIPT.md")
            (root / "blocked_step_audit.md").write_text("# blocked\n", encoding="utf-8")
            _write_diagnosis(
                root / "diagnosis.json",
                full=_mode_payload(tool_choice=57, oracle_choice=156, choose_swallow=0.60, family_mismatch=0.70, verify_usage=0.20),
                selector=_mode_payload(tool_choice=57, oracle_choice=156, choose_swallow=0.20, family_mismatch=0.20, verify_usage=0.40),
                voi=_mode_payload(tool_choice=57, oracle_choice=156, choose_swallow=0.40, family_mismatch=0.35, verify_usage=0.20),
                blocked_step_count=100,
            )

            summary = build_stress_probe_summary(run_root=root, delta=0.10)
            self.assertEqual("C", summary["route"])
            self.assertEqual("pause_for_gate_memory_selector_review", summary["next_action"])
            self.assertGreater(summary["negative_inversion_count"], 0)


if __name__ == "__main__":
    unittest.main()
