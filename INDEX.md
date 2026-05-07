# Public Artifact Index

This repository is the public code and artifact package linked from the CS505 final report.
All links below are repository-relative and should work on GitHub.

## Final Report

- [ACL PDF](report/final_report.pdf)
- [LaTeX source](report/final_report.tex)
- [BibTeX references](report/references.bib)
- [Figure generator](report/build_visual_assets.py)
- [Figures](report/figures/)

## Main Result Artifacts

- [v4 seed7 battery summary JSON](artifacts/logs/v4_seed7_live_battery_20260505/battery_summary_seed7.json)
- [v4 seed7 battery summary Markdown](artifacts/logs/v4_seed7_live_battery_20260505/battery_summary_seed7.md)
- [v4 reference receipt](artifacts/logs/v4_seed7_live_battery_20260505/reference_pack/POST_RUN_RECEIPT.md)
- [v4 stress receipt](artifacts/logs/v4_seed7_live_battery_20260505/stress_pack/POST_RUN_RECEIPT.md)
- [v4 reference diagnosis](artifacts/logs/v4_seed7_live_battery_20260505/reference_pack/diagnosis.json)
- [v4 stress diagnosis](artifacts/logs/v4_seed7_live_battery_20260505/stress_pack/diagnosis.json)
- [v4 reference blocked-step audit](artifacts/logs/v4_seed7_live_battery_20260505/reference_pack/blocked_step_audit.md)
- [v4 stress blocked-step audit](artifacts/logs/v4_seed7_live_battery_20260505/stress_pack/blocked_step_audit.md)
- [v4 pre-live gated summary](artifacts/logs/v4_seed7_live_battery_20260505/PRE_LIVE_GATED_SUMMARY.md)

## Offline Pack Repair Artifacts

- [v4 offline preflight summary JSON](artifacts/logs/v4_seed7_reference_repair_20260505/offline_preflight_summary.json)
- [v4 offline preflight summary Markdown](artifacts/logs/v4_seed7_reference_repair_20260505/offline_preflight_summary.md)
- [v4 repaired reference preflight](artifacts/logs/v4_seed7_reference_repair_20260505/attempt_01/task_pressure_preflight_summary.json)
- [v4 repaired reference tasks](artifacts/logs/v4_seed7_reference_repair_20260505/attempt_01/tasks_verify_necessary_reference_seed7_repair_attempt_01.jsonl)
- [v4 repaired reference SHA256](artifacts/logs/v4_seed7_reference_repair_20260505/attempt_01/tasks_verify_necessary_reference_seed7_repair_attempt_01.sha256.txt)
- [v4 regenerated stress preflight](artifacts/logs/v4_seed7_reference_repair_20260505/stress_pack/task_pressure_preflight_summary.json)
- [v4 regenerated stress tasks](artifacts/logs/v4_seed7_reference_repair_20260505/stress_pack/tasks_verify_necessary_aggressive_stress_v4_100_seed7.jsonl)
- [v4 regenerated stress SHA256](artifacts/logs/v4_seed7_reference_repair_20260505/stress_pack/tasks_verify_necessary_aggressive_stress_v4_100_seed7.sha256.txt)

## Seed11 Diagnostic Artifacts

- [seed11 stress battery summary JSON](artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/battery_summary_seed11.json)
- [seed11 stress battery summary Markdown](artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/battery_summary_seed11.md)
- [seed11 cross-seed summary JSON](artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/cross_seed_summary_v3.json)
- [seed11 cross-seed summary Markdown](artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/cross_seed_summary_v3.md)
- [seed11 stress receipt](artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/stress_pack/POST_RUN_RECEIPT.md)
- [seed11 stress diagnosis](artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/stress_pack/diagnosis.json)
- [seed11 stress blocked-step audit](artifacts/logs/seed11_v3_same_design_collapse_probe_20260423/stress_pack/blocked_step_audit.md)
- [seed11 reference task pack](artifacts/logs/seed11_v3_reference_confirmation_20260430/reference_pack/tasks_verify_necessary_reference_v1_100_seed11.jsonl)
- [seed11 reference preflight](artifacts/logs/seed11_v3_reference_confirmation_20260430/reference_pack/task_pressure_preflight_summary.json)

## Source Code

- [task generator](data/generate_tasks.py)
- [ToyTextMDP environment](env/toytextmdp.py)
- [suite runner](eval/run_suite.py)
- [receipt checker](eval/run_receipt.py)
- [diagnosis builder](eval/diagnose_reflection.py)
- [battery summary builder](eval/build_battery_summary.py)
- [VOI memory module](agent/voi_memory.py)
- [selector module](agent/selector.py)
- [agent loop](agent/loop.py)

## Tests

- [test suite directory](tests/)

## Reproduction Notes

Install the Python dependencies from [requirements.txt](requirements.txt), then run:

```powershell
python -m unittest discover tests
```

The final report itself is in [report/](report/). The public package intentionally contains a curated subset of the local workspace: code, selected logs, task packs, report sources, and generated figures.
