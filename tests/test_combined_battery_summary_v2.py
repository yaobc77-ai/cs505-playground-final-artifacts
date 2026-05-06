from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from eval.build_combined_battery_summary_v2 import build_combined_summary_v2


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
                "- run_name: synthetic",
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


class CombinedBatterySummaryV2Tests(unittest.TestCase):
    def test_stable_negative_result_is_detected(self) -> None:
        with _scratch_dir("combined_battery_stable_negative") as root:
            seed7_summary_path = root / "battery_summary_seed7.json"
            seed7_summary_path.write_text(
                json.dumps(
                    {
                        "seed11_trigger": False,
                        "next_action": "ready_for_seed11_same_design_collapse_probe_review",
                        "trigger_good": [],
                        "negative_inversion": [],
                        "stress": {"receipt_ok": True},
                    }
                ),
                encoding="utf-8",
            )
            seed11_root = root / "seed11_stress"
            seed11_root.mkdir()
            _write_receipt(seed11_root / "POST_RUN_RECEIPT.md")
            (seed11_root / "blocked_step_audit.md").write_text("# blocked\n", encoding="utf-8")
            _write_diagnosis(
                seed11_root / "diagnosis.json",
                full=_mode_payload(
                    tool_choice=100,
                    oracle_choice=200,
                    choose_swallow=0.50,
                    family_mismatch=0.50,
                    verify_usage=0.20,
                ),
                selector=_mode_payload(
                    tool_choice=100,
                    oracle_choice=200,
                    choose_swallow=0.50,
                    family_mismatch=0.50,
                    verify_usage=0.20,
                ),
                voi=_mode_payload(
                    tool_choice=100,
                    oracle_choice=200,
                    choose_swallow=0.50,
                    family_mismatch=0.50,
                    verify_usage=0.20,
                ),
                blocked_step_count=100,
            )

            summary = build_combined_summary_v2(
                seed7_battery_summary_path=seed7_summary_path,
                seed11_stress_run_root=seed11_root,
                delta=0.10,
            )
            self.assertTrue(summary["stable_negative_result"])
            self.assertEqual("enter_v3_aggressive_stress", summary["next_action"])
            self.assertEqual("stable_negative_result", summary["verdict"])


if __name__ == "__main__":
    unittest.main()
