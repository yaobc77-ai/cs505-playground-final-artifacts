from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from eval.build_cross_seed_probe_summary_v3 import build_cross_seed_probe_summary_v3


SCRATCH_ROOT = Path("C:/Users/yaobc/Documents/Playground/tests/_tmp_runtime")


@contextlib.contextmanager
def _scratch_dir(prefix: str):
    path = SCRATCH_ROOT / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class BuildCrossSeedProbeSummaryV3Tests(unittest.TestCase):
    def test_route_b_when_seed11_reopens_separation_without_inversion(self) -> None:
        with _scratch_dir("cross_seed_route_b") as root:
            seed7_summary = {
                "stress": {
                    "modes": {
                        "voi_memory_v1": {
                            "tool_choice_step_count": 100.0,
                            "oracle_tool_choice_step_count": 200.0,
                            "realized_tool_choice_coverage": 0.5,
                            "choose_swallow_rate_given_oracle_tool_choice": 0.5,
                            "selected_vs_oracle_family_mismatch_rate": 0.5,
                            "tool_family_mismatch_rate_vs_oracle": 0.0,
                            "verify_usage_rate": 0.125,
                            "wrong_tool_rate": 0.0,
                        }
                    }
                }
            }
            seed11_summary = {
                "stress_pack": {
                    "receipt_ok": True,
                    "modes": {
                        "voi_memory_v1": {
                            "tool_choice_step_count": 83.0,
                            "oracle_tool_choice_step_count": 183.0,
                            "realized_tool_choice_coverage": 83.0 / 183.0,
                            "choose_swallow_rate_given_oracle_tool_choice": 0.5464,
                            "selected_vs_oracle_family_mismatch_rate": 0.5464,
                            "tool_family_mismatch_rate_vs_oracle": 0.0,
                            "verify_usage_rate": 0.0984,
                            "wrong_tool_rate": 0.0,
                        }
                    },
                },
                "pure_task_pressure": True,
                "negative_inversion_count": 0,
                "separation_reopened": True,
                "trigger_good_count": 1,
            }
            seed7_path = root / "seed7.json"
            seed11_path = root / "seed11.json"
            seed7_path.write_text(json.dumps(seed7_summary), encoding="utf-8")
            seed11_path.write_text(json.dumps(seed11_summary), encoding="utf-8")

            summary = build_cross_seed_probe_summary_v3(
                seed7_battery_summary_path=seed7_path,
                seed11_stress_summary_path=seed11_path,
                delta=0.10,
            )
            self.assertTrue(summary["seed11_receipt_ok"])
            self.assertTrue(summary["seed11_choose_swallow_collapse_shape_persists"])
            self.assertTrue(summary["seed11_pure_task_pressure"])
            self.assertEqual(0, summary["seed11_negative_inversion_count"])
            self.assertEqual("B", summary["route"])
            self.assertEqual("prepare_seed11_reference_confirmation_review", summary["next_action"])

    def test_route_a_when_second_seed_collapses_again(self) -> None:
        with _scratch_dir("cross_seed_route_a") as root:
            seed7_summary = {
                "stress": {
                    "modes": {
                        "voi_memory_v1": {
                            "tool_choice_step_count": 100.0,
                            "oracle_tool_choice_step_count": 200.0,
                            "realized_tool_choice_coverage": 0.5,
                            "choose_swallow_rate_given_oracle_tool_choice": 0.5,
                            "selected_vs_oracle_family_mismatch_rate": 0.5,
                            "tool_family_mismatch_rate_vs_oracle": 0.0,
                            "verify_usage_rate": 0.125,
                            "wrong_tool_rate": 0.0,
                        }
                    }
                }
            }
            seed11_summary = {
                "stress_pack": {
                    "receipt_ok": True,
                    "modes": {
                        "voi_memory_v1": {
                            "tool_choice_step_count": 100.0,
                            "oracle_tool_choice_step_count": 200.0,
                            "realized_tool_choice_coverage": 0.5,
                            "choose_swallow_rate_given_oracle_tool_choice": 0.53,
                            "selected_vs_oracle_family_mismatch_rate": 0.52,
                            "tool_family_mismatch_rate_vs_oracle": 0.0,
                            "verify_usage_rate": 0.12,
                            "wrong_tool_rate": 0.0,
                        }
                    },
                },
                "pure_task_pressure": True,
                "negative_inversion_count": 0,
                "separation_reopened": False,
                "trigger_good_count": 0,
            }
            seed7_path = root / "seed7.json"
            seed11_path = root / "seed11.json"
            seed7_path.write_text(json.dumps(seed7_summary), encoding="utf-8")
            seed11_path.write_text(json.dumps(seed11_summary), encoding="utf-8")

            summary = build_cross_seed_probe_summary_v3(
                seed7_battery_summary_path=seed7_path,
                seed11_stress_summary_path=seed11_path,
                delta=0.10,
            )
            self.assertEqual("A", summary["route"])
            self.assertEqual("enter_stronger_v3_patch", summary["next_action"])


if __name__ == "__main__":
    unittest.main()
