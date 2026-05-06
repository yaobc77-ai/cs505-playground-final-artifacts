# Pre-Live Gated Summary: v4 seed7 dual-battery live

- date: 2026-05-06T01:46:05.423501+00:00
- lane: discovery
- action: v4_seed7_dual_battery_live
- seed: 7
- modes: full -> selector_penalty -> voi_memory_v1
- reference_run_root: <PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\reference_pack
- stress_run_root: <PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\stress_pack
- allow_preflight_failure: false
- task_regeneration: false
- gate_memory_selector_loop_edits: false

## Frozen Input Checks
- reference.task_exists: true
- reference.sha_file_exists: true
- reference.preflight_exists: true
- reference.sha256: f120c03abaa18ace347466d39b7666cc2850535d424d07bf3fb7572e86a8e10e
- reference.sha_file_matches: true
- reference.combined_preflight_passes: true
- stress.task_exists: true
- stress.sha_file_exists: true
- stress.preflight_exists: true
- stress.sha256: b476cb1dc79e59aac84710cfae9428be791c0a652e8831fe1ef34ff6161b9da4
- stress.sha_file_matches: true
- stress.combined_preflight_passes: true

- gated_summary_status: pass
