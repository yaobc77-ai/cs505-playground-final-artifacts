# Battery Summary

- delta: 0.10
- route: B
- seed11_trigger: false
- next_action: v4_task_pressure_insufficient_or_stable_negative_review

## Receipt Layer

- reference.receipt_ok: true
- stress.receipt_ok: true

## Trigger Good

- none
- trigger_good_primary_mode: voi_memory_v1
- trigger_good_primary_count: 0

## Sanity Check Deltas

- full / verify_usage_rate / higher_than_reference / reference=0.1034 / stress=0.2980 / delta=0.1946
- selector_penalty / verify_usage_rate / higher_than_reference / reference=0.0364 / stress=0.1736 / delta=0.1372

## Wrong Tool Surface Exposed

- full / reference=0.0000 / stress=0.2649
- selector_penalty / reference=0.0000 / stress=0.1488

## Negative Inversion

- none
- negative_inversion_count: 0

## Reference Pack

- run_name: reference_pack
- run_root: <PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\reference_pack
- blocked_step: task_pressure=100, surfaced_ask_deficit=0, hard_cutoff_candidate=0, route_target=task_pressure

| mode | tool_choice_step_count | oracle_tool_choice_step_count | realized_tool_choice_coverage | choose_swallow | selected_vs_oracle_family_mismatch | tool_family_mismatch | verify_usage_rate | wrong_tool_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 45.0 | 145.0 | 0.3103 | 0.6897 | 0.6897 | 0.0000 | 0.1034 | 0.0000 |
| selector_penalty | 10.0 | 110.0 | 0.0909 | 0.9091 | 0.9091 | 0.0000 | 0.0364 | 0.0000 |
| voi_memory_v1 | 89.0 | 189.0 | 0.4709 | 0.5291 | 0.5291 | 0.0000 | 0.2063 | 0.0000 |

## Stress Pack

- run_name: stress_pack
- run_root: <PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\stress_pack
- blocked_step: task_pressure=100, surfaced_ask_deficit=0, hard_cutoff_candidate=0, route_target=task_pressure

| mode | tool_choice_step_count | oracle_tool_choice_step_count | realized_tool_choice_coverage | choose_swallow | selected_vs_oracle_family_mismatch | tool_family_mismatch | verify_usage_rate | wrong_tool_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 51.0 | 151.0 | 0.3377 | 0.6623 | 0.9272 | 0.2649 | 0.2980 | 0.0000 |
| selector_penalty | 21.0 | 121.0 | 0.1736 | 0.8264 | 0.9752 | 0.1488 | 0.1736 | 0.0000 |
| voi_memory_v1 | 86.0 | 186.0 | 0.4624 | 0.5376 | 0.5376 | 0.0000 | 0.0860 | 0.0000 |
