# CS505 Playground Final Report

This repository contains the public code and curated artifacts for the CS505 final report:

**Making Tool Choice Observable: A Receipt-Safe Diagnostic Study of VOI-Based Memory in a Toolized Text Environment**

## Contents

- `agent/`, `data/`, `env/`, `eval/`, `tests/`: source code used for task generation, the ToyTextMDP environment, live runs, receipts, diagnosis, and report metrics.
- `report/`: ACL LaTeX source, compiled PDF, bibliography, ACL style files, deterministic figure generator, and generated figures.
- `artifacts/`: curated evidence artifacts for the reported seed-11 and v4 seed-7 batteries, including summaries, preflight summaries, receipts, diagnosis reports, blocked-step audits, task packs, and SHA files where applicable.
- `requirements.txt`: Python dependencies used by the project.

The repository intentionally excludes unrelated local workspace folders, local Codex state, broad raw live JSONL traces, and unrelated course projects. The included artifacts are the public subset needed to inspect the paper's reported claims without publishing the full local workspace.

## Main Report

The compiled report is:

```text
report/final_report.pdf
```

The LaTeX source is:

```text
report/final_report.tex
```

## Rebuild The Figures

From the repository root:

```bash
python report/build_visual_assets.py
```

## Run Tests

From the repository root:

```bash
python -m unittest discover tests
```

## Key Evidence Mapping

- Final v4 seed-7 live battery:
  - `artifacts/logs/v4_seed7_live_battery_20260505/battery_summary_seed7.json`
  - `artifacts/logs/v4_seed7_live_battery_20260505/battery_summary_seed7.md`
  - `artifacts/logs/v4_seed7_live_battery_20260505/reference_pack/POST_RUN_RECEIPT.md`
  - `artifacts/logs/v4_seed7_live_battery_20260505/stress_pack/POST_RUN_RECEIPT.md`

- v4 seed-7 offline repair/preflight:
  - `artifacts/logs/v4_seed7_reference_repair_20260505/offline_preflight_summary.json`
  - `artifacts/logs/v4_seed7_reference_repair_20260505/offline_preflight_summary.md`

- Seed-11 confirmation/probe context:
  - `artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/battery_summary_seed11.json`
  - `artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/battery_summary_seed11.md`
  - `artifacts/logs/seed11_v3_reference_confirmation_20260430/reference_pack/task_pressure_preflight_summary.json`

## Claim Boundary

The final report does not claim stable which-tool improvement. It reports receipt-safe negative evidence for `voi_memory_v1` under the current task ecology, while preserving the narrower earlier claim that the method improved whether-to-ask behavior in a single-tool setting.
