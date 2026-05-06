# Seed11 Stress Probe Summary

- delta: 0.10
- verdict: separation_reopened
- route: B
- next_action: prepare_seed11_reference_confirmation_review
- receipt_ok: true
- pure_task_pressure: true
- separation_reopened: true
- negative_inversion_count: 0

## Trigger Good

- selected_vs_oracle_family_mismatch_rate / voi_memory_v1_lower_than_full / full=0.8397 / voi_memory_v1=0.5464 / delta=0.2933

## Negative Inversion

- none

## Stress Pack Metrics

- run_name: stress_pack
- run_root: <PLAYGROUND_ROOT>\logs\seed11_v3_same_design_collapse_probe_20260423\stress_pack
- blocked_step: task_pressure=100, surfaced_ask_deficit=0, hard_cutoff_candidate=0, route_target=task_pressure

| mode | tool_choice_step_count | oracle_tool_choice_step_count | realized_tool_choice_coverage | choose_swallow | selected_vs_oracle_family_mismatch | tool_family_mismatch | verify_usage_rate | wrong_tool_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 57.0 | 156.0 | 0.3654 | 0.6346 | 0.8397 | 0.2051 | 0.3057 | 0.0000 |
| selector_penalty | 29.0 | 129.0 | 0.2248 | 0.7752 | 0.9225 | 0.1473 | 0.1938 | 0.0000 |
| voi_memory_v1 | 83.0 | 183.0 | 0.4536 | 0.5464 | 0.5464 | 0.0000 | 0.0984 | 0.0000 |
