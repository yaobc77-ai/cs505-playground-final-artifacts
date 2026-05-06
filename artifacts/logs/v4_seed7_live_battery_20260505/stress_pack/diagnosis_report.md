# Reflection Failure-Mode Diagnosis

Run root: `<PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\stress_pack`
Seeds: 0

## Full Episode Summary

- Success rate: 0.810
- Avg reward: 0.768
- Avg steps: 1.510
- Avg asks: 0.510

## Full vs No Reflection

- Full regretful ASK rate: 0.157
- No reflection regretful ASK rate: n/a
- Full choose regret rate: 0.140
- No reflection choose regret rate: n/a
- Full selected ASK rate: 0.338
- No reflection selected ASK rate: n/a
- Full over-ask cost per step: 0.003
- No reflection over-ask cost per step: n/a
- Full under-ask cost per step: 0.084
- No reflection under-ask cost per step: n/a
- Full choose-without-prior-ask rate: 0.520
- No reflection choose-without-prior-ask rate: n/a

## Tool-Use Summary

- Full scan usage rate: 0.040
- No reflection scan usage rate: n/a
- Full verify usage rate: 0.298
- No reflection verify usage rate: n/a
- Full tool precision: 1.000 (tool_choice_step_count=51)
- No reflection tool precision: n/a (tool_choice_step_count=0)
- Full wrong-tool rate: 0.000
- No reflection wrong-tool rate: n/a
- Full over-tool cost per step: 0.000
- No reflection over-tool cost per step: n/a
- Full under-tool cost per step: 0.000
- No reflection under-tool cost per step: n/a

## Runtime Tool Diagnosis

- Full candidate backend template rate: 0.026
- Full candidate backend union rate: 0.000
- Full oracle tool-choice step count (ask-opportunity proxy): 151.000
- Full oracle family gap mean: 0.212
- Full final oracle family alignment rate: 0.556
- Full final oracle scan coverage rate: 0.046
- Full final oracle verify coverage rate: 0.550
- Full selector family override rate: 0.000
- No reflection oracle tool-choice step count (ask-opportunity proxy): n/a
- No reflection final oracle family alignment rate: n/a
- No reflection selector family override rate: n/a

## Candidate-Layer Diagnosis

### full

- Selected-vs-oracle mismatch rate: 0.927
- Choose-swallow rate: 0.662
- Surface-family distortion rate: 0.265
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.308
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 1.000
- Best-verify missing reasons: not_generated=0.119, generated_but_pruned=0.000, generated_noncanonical=0.000, dominated_after_scoring=0.550
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

### selector_penalty

- Selected-vs-oracle mismatch rate: 0.975
- Choose-swallow rate: 0.826
- Surface-family distortion rate: 0.149
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.358
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 1.000
- Best-verify missing reasons: not_generated=0.008, generated_but_pruned=0.000, generated_noncanonical=0.000, dominated_after_scoring=0.744
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

### voi_memory_v1

- Selected-vs-oracle mismatch rate: 0.538
- Choose-swallow rate: 0.538
- Surface-family distortion rate: 0.000
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.296
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 0.462
- Best-verify missing reasons: not_generated=0.000, generated_but_pruned=0.538, generated_noncanonical=0.000, dominated_after_scoring=0.376
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

## Verify Lineage Survival Audit

### full

- Verify lineage count: 851
- Generated: denominator=851, applicable=851, retained=140, retained_rate=0.165
- Family merge: denominator=851, applicable=140, retained=140, retained_rate=1.000
- Normalization merge: denominator=851, applicable=140, retained=140, retained_rate=1.000
- Dedup: denominator=851, applicable=140, retained=140, retained_rate=1.000
- Candidate cap (source-side): denominator=851, applicable=851, retained=140, retained_rate=0.165
- Pre-selector filter: denominator=851, applicable=158, retained=158, retained_rate=1.000
- Selector entry: denominator=851, applicable=158, retained=158, retained_rate=1.000
- Retained before selector rate: 0.186
- Injected before selector: 0.021
- Shape replacement detected: 0.121
- Pruned stage distribution: generated=0.579, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.235, pre_selector_filter=0.000, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.024, deduplicated=0.000, budget_cap=0.211, filter_rule=0.000, unknown=0.579

### selector_penalty

- Verify lineage count: 705
- Generated: denominator=705, applicable=705, retained=124, retained_rate=0.176
- Family merge: denominator=705, applicable=124, retained=124, retained_rate=1.000
- Normalization merge: denominator=705, applicable=124, retained=124, retained_rate=1.000
- Dedup: denominator=705, applicable=124, retained=124, retained_rate=1.000
- Candidate cap (source-side): denominator=705, applicable=705, retained=124, retained_rate=0.176
- Pre-selector filter: denominator=705, applicable=125, retained=125, retained_rate=1.000
- Selector entry: denominator=705, applicable=125, retained=125, retained_rate=1.000
- Retained before selector rate: 0.177
- Injected before selector: 0.001
- Shape replacement detected: 0.115
- Pruned stage distribution: generated=0.630, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.193, pre_selector_filter=0.000, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.000, deduplicated=0.000, budget_cap=0.193, filter_rule=0.000, unknown=0.630

### voi_memory_v1

- Verify lineage count: 1030
- Generated: denominator=1030, applicable=1030, retained=370, retained_rate=0.359
- Family merge: denominator=1030, applicable=370, retained=370, retained_rate=1.000
- Normalization merge: denominator=1030, applicable=370, retained=370, retained_rate=1.000
- Dedup: denominator=1030, applicable=370, retained=370, retained_rate=1.000
- Candidate cap (source-side): denominator=1030, applicable=1030, retained=370, retained_rate=0.359
- Pre-selector filter: denominator=1030, applicable=370, retained=172, retained_rate=0.465
- Selector entry: denominator=1030, applicable=370, retained=172, retained_rate=0.465
- Retained before selector rate: 0.359
- Injected before selector: 0.000
- Shape replacement detected: 0.156
- Pruned stage distribution: generated=0.641, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.000, pre_selector_filter=0.192, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.000, deduplicated=0.000, budget_cap=0.000, filter_rule=0.192, unknown=0.641

## Pre-Selector Pruning Attribution

- Claim boundary: This audit attributes which gate-facing reason accounted for the block; it does not recommend changing the gate.
- Primary evidence fields:
  - verify_pruned_stage
  - verify_pruned_reason
  - verify_retained_before_selector
  - verify_entered_selector
  - ask_gate_enabled
  - ask_gate_allowed
  - ask_gate_reason
  - pre_selector_block_source
  - pre_selector_block_reason
- Secondary context fields:
  - ask_gate_score
  - ask_advantage
  - best_ask_family
  - best_ask_eval
  - best_choose_eval
  - voi_ratio
  - uncertainty
  - base_uncertainty
  - reliability_uncertainty
  - evidence_conflict
  - memory_signal
  - helpful_ask
  - choose_regret
  - regretful_ask
  - choose_helpful
- `voi_memory_v1` pre-selector summary: denominator=1030, applicable_count=370, retained_count=172, retained_rate=0.465
- Filter-rule by ask_gate_reason: strong_negative_voi=198
- Filter-rule by tool_subtype: deceptive_scan=122, scan_then_verify=40, verify_first=36
- Verify vs scan pre-selector survival:
  - verify: applicable_count=370, retained_count=172, retained_rate=0.465
  - scan: applicable_count=411, retained_count=202, retained_rate=0.491
- Direct gate evidence coverage: filter_rule_without_direct_gate_evidence=0

## Blocked ASK Value Audit

- Claim boundary: This audit classifies blocked strong_negative_voi steps into task pressure, surfaced ask deficit, or hard cutoff candidates; it does not recommend changing the gate.
- Blocked step count: 100
- task_pressure: count=100, rate=1.000
- surfaced_ask_deficit: count=0, rate=0.000
- hard_cutoff_candidate: count=0, rate=0.000
- Direct choose correct rate: 1.000
- Surface gap distribution: count=100, mean=0.050, median=0.030, positive_rate=0.700
- Suggested next route: task_pressure
- Requires new experiment: false

## Bottleneck Routing

- Routing modes: full, selector_penalty, voi_memory_v1
- Most likely layer: candidate_coverage
- Candidate-related share: 1.000
- Policy-family error share: 0.000

## selector_penalty Snapshot

- Success rate: 0.880
- Avg reward: 0.861
- Avg steps: 1.210
- Avg asks: 0.210
- Regretful ASK rate: 0.048
- Choose regret rate: 0.110
- Over-ask cost per step: 0.000
- Under-ask cost per step: 0.084
- Scan usage rate: 0.000
- Verify usage rate: 0.174
- Tool precision: 1.000 (tool_choice_step_count=21)
- Wrong-tool rate: 0.000
- Over-tool cost per step: 0.000
- Under-tool cost per step: 0.000
- Candidate backend template rate: 0.017
- Candidate backend union rate: 0.000
- Oracle tool-choice step count (ask-opportunity proxy): 121.000
- Oracle family gap mean: 0.205
- Final oracle family alignment rate: 0.355
- Final oracle scan coverage rate: 0.025
- Final oracle verify coverage rate: 0.719
- Selector family override rate: 0.000

## voi_memory_v1 Snapshot

- Success rate: 1.000
- Avg reward: 0.965
- Avg steps: 1.860
- Avg asks: 0.860
- Regretful ASK rate: 0.000
- Choose regret rate: 0.000
- Over-ask cost per step: 0.000
- Under-ask cost per step: 0.000
- Scan usage rate: 0.376
- Verify usage rate: 0.086
- Tool precision: 1.000 (tool_choice_step_count=86)
- Wrong-tool rate: 0.000
- Over-tool cost per step: 0.000
- Under-tool cost per step: 0.000
- Candidate backend template rate: 0.011
- Candidate backend union rate: 0.989
- Oracle tool-choice step count (ask-opportunity proxy): 186.000
- Oracle family gap mean: 0.153
- Final oracle family alignment rate: 0.629
- Final oracle scan coverage rate: 0.441
- Final oracle verify coverage rate: 0.871
- Selector family override rate: 0.000
- Ask-gate block rate: 0.538
- Ask-gate allow rate given positive ask advantage: 1.000
- Ask-gate block rate given nonpositive ask advantage: 1.000
- Retrieved positive evidence rate: 0.989
- Retrieved caution evidence rate: 0.995

## Difficulty Breakdown

| Group | Full regretful ASK | No reflection regretful ASK | Full choose regret | No reflection choose regret | Full over-ask cost | Full under-ask cost |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 0.067 | n/a | 0.086 | n/a | 0.001 | 0.058 |
| hard | 0.267 | n/a | 0.227 | n/a | 0.004 | 0.116 |
| medium | 0.143 | n/a | 0.140 | n/a | 0.003 | 0.086 |

## Noise Breakdown

| Group | Full regretful ASK | No reflection regretful ASK | Full choose regret | No reflection choose regret | Full over-ask cost | Full under-ask cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.067 | n/a | 0.086 | n/a | 0.001 | 0.058 |
| 0.18 | 0.143 | n/a | 0.140 | n/a | 0.003 | 0.086 |
| 0.32 | 0.267 | n/a | 0.227 | n/a | 0.004 | 0.116 |

## Reflection Avoid-Action Distribution

- avoid_actions=['CHOOSE']: 0.470 (47)
- avoid_actions=['ASK']: 0.410 (41)
- avoid_actions=[]: 0.070 (7)
- mixed avoid actions: 0.050 (5)

## Reflection Delta Summary

- info_seeking: mean=0.096, abs_mean=0.264, positive=0.510, negative=0.420
- cost_sensitivity: mean=0.094, abs_mean=0.140, positive=0.460, negative=0.120
- confidence_threshold: mean=0.170, abs_mean=0.211, positive=0.560, negative=0.170

## Subsequent Correlation Summary

- avoid CHOOSE -> next ask count: -0.097
- avoid CHOOSE -> next reward: 0.042
- confidence threshold delta -> next ask count: -0.011
- info seeking delta -> next ask count: -0.058

## Memory-Mediated CHOOSE Suppression

- Selected ASK with retrieved CHOOSE-suppressing memory: 0.980
- Regretful ASK with CHOOSE-suppressing memory: 0.875

## Under-asking Profile

- Full choose regret rate: 0.140
- Full under-ask cost per choose regret: 0.907
- Full choose regret without prior ask rate: 0.929
- Full choose regret high-margin share: 0.000

## Recommendation

- Recommended next intervention: `voi_memory_v1`
- voi_memory_v1 has the strongest experimental profile: avg_reward=0.965, success_rate=1.000.
- voi_memory_v1 keeps regretful_ask_rate=0.000 and choose_regret_rate=0.000.
- voi_memory_v1 keeps over_ask_cost_per_step=0.000 and under_ask_cost_per_step=0.000.
- voi_memory_v1 keeps tool_precision=1.000, wrong_tool_rate=0.000, over_tool_cost_per_step=0.000, under_tool_cost_per_step=0.000.
- Compared with full: reward delta=0.197, success delta=0.190.
