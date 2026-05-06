# Blocked ASK Value Audit

Run root: `<PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\reference_pack`
Seeds: 0

## Focus

- Current focus: `blocked_ask_value_audit`
- Execution lane: `discovery`
- Claim boundary: This audit classifies blocked strong_negative_voi steps into task pressure, surfaced ask deficit, or hard cutoff candidates; it does not recommend changing the gate.
- Scope: classify step-level strong_negative_voi blocks on the existing live seed-7 run; do not infer a gate change from this audit alone.

## `voi_memory_v1` Strong-Negative Blocks

- Blocked step count: 100
- task_pressure: count=100, rate=1.000
- surfaced_ask_deficit: count=0, rate=0.000
- hard_cutoff_candidate: count=0, rate=0.000
- Direct choose correct rate: 1.000
- Best surfaced ASK missing count: 0
- Surface gap distribution: count=100, mean=0.000, median=0.000, positive_rate=0.000
- Routing threshold: 0.600
- Suggested next route: task_pressure
- Requires new experiment: false