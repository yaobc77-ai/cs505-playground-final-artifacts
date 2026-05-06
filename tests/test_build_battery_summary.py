from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from eval.build_battery_summary import build_seed_battery_summary, render_seed_battery_summary


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
    wrong_tool: float = 0.0,
    tool_family_mismatch: float = 0.0,
) -> dict[str, object]:
    return {
        "overall": {
            "tool_choice_step_count": tool_choice,
            "oracle_tool_choice_step_count": oracle_choice,
            "choose_swallow_rate_given_oracle_tool_choice": choose_swallow,
            "selected_vs_oracle_family_mismatch_rate": family_mismatch,
            "verify_usage_rate": verify_usage,
            "wrong_tool_rate": wrong_tool,
            "tool_family_mismatch_rate_vs_oracle": tool_family_mismatch,
        },
        "episodes": {
            "success_rate": 1.0,
        },
    }


def _write_diagnosis(path: Path, *, full: dict[str, object], selector: dict[str, object], voi: dict[str, object]) -> None:
    payload = {
        "modes": {
            "full": full,
            "selector_penalty": selector,
            "voi_memory_v1": voi,
        },
        "blocked_step_audit": {
            "voi_memory_v1": {
                "task_pressure_count": 10,
                "surfaced_ask_deficit_count": 0,
                "hard_cutoff_candidate_count": 0,
                "blocked_step_count": 10,
                "route_target": "task_pressure",
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BuildBatterySummaryTests(unittest.TestCase):
    def test_trigger_good_uses_reference_pack_same_mode_baseline_for_voi_primary_mode(self) -> None:
        with _scratch_dir("battery_summary_reference_baseline") as root:
            reference_root = root / "reference"
            stress_root = root / "stress"
            reference_root.mkdir()
            stress_root.mkdir()

            for target in (reference_root, stress_root):
                _write_receipt(target / "POST_RUN_RECEIPT.md")
                (target / "blocked_step_audit.md").write_text("# blocked\n", encoding="utf-8")

            _write_diagnosis(
                reference_root / "diagnosis.json",
                full=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.60,
                    family_mismatch=0.55,
                    verify_usage=0.20,
                ),
                selector=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.55,
                    family_mismatch=0.50,
                    verify_usage=0.25,
                ),
                voi=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.58,
                    family_mismatch=0.52,
                    verify_usage=0.22,
                ),
            )
            _write_diagnosis(
                stress_root / "diagnosis.json",
                full=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.45,
                    family_mismatch=0.40,
                    verify_usage=0.35,
                ),
                selector=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.50,
                    family_mismatch=0.44,
                    verify_usage=0.28,
                ),
                voi=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.44,
                    family_mismatch=0.39,
                    verify_usage=0.34,
                ),
            )

            summary = build_seed_battery_summary(
                reference_run_root=reference_root,
                stress_run_root=stress_root,
                delta=0.10,
            )
            trigger_metrics = {(item["mode"], item["metric"]) for item in summary["trigger_good"]}
            self.assertEqual(
                {
                    ("voi_memory_v1", "choose_swallow_rate_given_oracle_tool_choice"),
                    ("voi_memory_v1", "selected_vs_oracle_family_mismatch_rate"),
                    ("voi_memory_v1", "verify_usage_rate"),
                },
                trigger_metrics,
            )
            sanity_metrics = {(item["mode"], item["metric"]) for item in summary["sanity_check_deltas"]}
            self.assertIn(("full", "choose_swallow_rate_given_oracle_tool_choice"), sanity_metrics)
            self.assertNotIn(("selector_penalty", "verify_usage_rate"), sanity_metrics)
            self.assertEqual("voi_memory_v1", summary["trigger_good_primary_mode"])
            self.assertEqual(3, summary["trigger_good_primary_count"])
            self.assertEqual("A", summary["route"])
            self.assertTrue(summary["seed11_trigger"])
            self.assertEqual("seed13_mixed_battery_larger_confirmation", summary["next_action"])
            rendered = render_seed_battery_summary(summary)
            self.assertIn("route: A", rendered)
            self.assertIn("trigger_good_primary_mode: voi_memory_v1", rendered)
            self.assertIn("reference=0.5800 / stress=0.4400", rendered)

    def test_negative_inversion_routes_to_v3_review(self) -> None:
        with _scratch_dir("battery_summary_negative_inversion") as root:
            reference_root = root / "reference"
            stress_root = root / "stress"
            reference_root.mkdir()
            stress_root.mkdir()

            for target in (reference_root, stress_root):
                _write_receipt(target / "POST_RUN_RECEIPT.md")
                (target / "blocked_step_audit.md").write_text("# blocked\n", encoding="utf-8")

            _write_diagnosis(
                reference_root / "diagnosis.json",
                full=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.50,
                    family_mismatch=0.50,
                    verify_usage=0.30,
                ),
                selector=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.48,
                    family_mismatch=0.48,
                    verify_usage=0.32,
                ),
                voi=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.47,
                    family_mismatch=0.47,
                    verify_usage=0.33,
                ),
            )
            _write_diagnosis(
                stress_root / "diagnosis.json",
                full=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.42,
                    family_mismatch=0.42,
                    verify_usage=0.35,
                ),
                selector=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.20,
                    family_mismatch=0.18,
                    verify_usage=0.55,
                ),
                voi=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.35,
                    family_mismatch=0.31,
                    verify_usage=0.40,
                ),
            )

            summary = build_seed_battery_summary(
                reference_run_root=reference_root,
                stress_run_root=stress_root,
                delta=0.10,
            )
            self.assertFalse(summary["seed11_trigger"])
            self.assertEqual("C", summary["route"])
            self.assertEqual("ask_gate_memory_selector_diagnosis", summary["next_action"])
            metrics = {item["metric"] for item in summary["negative_inversion"]}
            self.assertEqual(
                {
                    "choose_swallow_rate_given_oracle_tool_choice",
                    "selected_vs_oracle_family_mismatch_rate",
                    "verify_usage_rate",
                },
                metrics,
            )
            self.assertEqual(3, summary["negative_inversion_count"])

    def test_reference_confirmation_eaten_by_reference_routes_to_v4_design(self) -> None:
        with _scratch_dir("battery_summary_reference_eats_signal") as root:
            reference_root = root / "reference"
            stress_root = root / "stress"
            reference_root.mkdir()
            stress_root.mkdir()

            for target in (reference_root, stress_root):
                _write_receipt(target / "POST_RUN_RECEIPT.md")
                (target / "blocked_step_audit.md").write_text("# blocked\n", encoding="utf-8")

            _write_diagnosis(
                reference_root / "diagnosis.json",
                full=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.60,
                    family_mismatch=0.60,
                    verify_usage=0.20,
                ),
                selector=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.75,
                    family_mismatch=0.75,
                    verify_usage=0.15,
                ),
                voi=_mode_payload(
                    tool_choice=20,
                    oracle_choice=40,
                    choose_swallow=0.58,
                    family_mismatch=0.58,
                    verify_usage=0.22,
                ),
            )
            _write_diagnosis(
                stress_root / "diagnosis.json",
                full=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.62,
                    family_mismatch=0.62,
                    verify_usage=0.25,
                ),
                selector=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.76,
                    family_mismatch=0.76,
                    verify_usage=0.18,
                ),
                voi=_mode_payload(
                    tool_choice=24,
                    oracle_choice=40,
                    choose_swallow=0.54,
                    family_mismatch=0.54,
                    verify_usage=0.20,
                ),
            )

            summary = build_seed_battery_summary(
                reference_run_root=reference_root,
                stress_run_root=stress_root,
                delta=0.10,
            )
            self.assertEqual([], summary["trigger_good"])
            self.assertEqual(0, summary["negative_inversion_count"])
            self.assertEqual("B", summary["route"])
            self.assertEqual("enter_v4_aggressive_stress_design", summary["next_action"])

    def test_route_b_next_action_can_be_overridden_for_v4_followup(self) -> None:
        with _scratch_dir("battery_summary_v4_route_b") as root:
            reference_root = root / "reference"
            stress_root = root / "stress"
            reference_root.mkdir()
            stress_root.mkdir()

            for target in (reference_root, stress_root):
                _write_receipt(target / "POST_RUN_RECEIPT.md")
                (target / "blocked_step_audit.md").write_text("# blocked\n", encoding="utf-8")

            flat_mode = _mode_payload(
                tool_choice=20,
                oracle_choice=40,
                choose_swallow=0.50,
                family_mismatch=0.50,
                verify_usage=0.20,
            )
            _write_diagnosis(
                reference_root / "diagnosis.json",
                full=flat_mode,
                selector=flat_mode,
                voi=flat_mode,
            )
            _write_diagnosis(
                stress_root / "diagnosis.json",
                full=flat_mode,
                selector=flat_mode,
                voi=flat_mode,
            )

            summary = build_seed_battery_summary(
                reference_run_root=reference_root,
                stress_run_root=stress_root,
                delta=0.10,
                route_b_next_action="v4_task_pressure_insufficient_or_stable_negative_review",
            )
            self.assertEqual("B", summary["route"])
            self.assertEqual(
                "v4_task_pressure_insufficient_or_stable_negative_review",
                summary["next_action"],
            )


if __name__ == "__main__":
    unittest.main()
