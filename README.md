# CS505 Playground Final Report

This repository contains the public code and selected artifacts for the CS505 final report:

**Making Tool Choice Observable: A Receipt-Safe Diagnostic Study of VOI-Based Memory in a Toolized Text Environment**

## Contents

| Path | Description |
| --- | --- |
| `agent/` | Agent components, including memory, selector, and loop code. |
| `data/` | Task generation code and task files. |
| `env/` | ToyTextMDP environment implementation. |
| `eval/` | Evaluation, receipt, diagnosis, and summary scripts. |
| `tests/` | Unit tests for the public package. |
| `report/` | ACL source, compiled PDF, bibliography, figures, and figure-generation script. |
| `artifacts/` | Curated summaries, preflight outputs, receipts, diagnosis files, blocked-step audits, task packs, and SHA files. |
| `requirements.txt` | Python dependency list. |

The repository is a curated public subset of the local project workspace. It includes the code and artifacts needed to inspect the report's claims, while leaving out unrelated local files, editor/tool state, broad raw traces, and unrelated course work.

## Main Report

The compiled report is:

```text
report/final_report.pdf
```

The LaTeX source is:

```text
report/final_report.tex
```

## Rebuild Figures

From the repository root:

```bash
python report/build_visual_assets.py
```

The checked-in PNG figures and compiled PDF are the submitted artifacts. Rebuilding figures on a non-Windows machine may use Pillow font fallback and slightly change text wrapping.

## Run Tests

From the repository root:

```bash
python -m unittest discover tests
```

## Key Evidence Mapping

Final v4 seed-7 live battery:

- `artifacts/logs/v4_seed7_live_battery_20260505/battery_summary_seed7.json`
- `artifacts/logs/v4_seed7_live_battery_20260505/battery_summary_seed7.md`
- `artifacts/logs/v4_seed7_live_battery_20260505/reference_pack/POST_RUN_RECEIPT.md`
- `artifacts/logs/v4_seed7_live_battery_20260505/stress_pack/POST_RUN_RECEIPT.md`

V4 seed-7 offline repair and preflight:

- `artifacts/logs/v4_seed7_reference_repair_20260505/offline_preflight_summary.json`
- `artifacts/logs/v4_seed7_reference_repair_20260505/offline_preflight_summary.md`

Seed-11 confirmation and probe context:

- `artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/battery_summary_seed11.json`
- `artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/battery_summary_seed11.md`
- `artifacts/logs/seed11_v3_reference_confirmation_20260430/reference_pack/task_pressure_preflight_summary.json`

## Claim Boundary

The final report does not claim stable which-tool improvement. It reports receipt-safe negative evidence for `voi_memory_v1` under the current task ecology, while preserving the narrower earlier claim that the method improved whether-to-ask behavior in a single-tool setting.
