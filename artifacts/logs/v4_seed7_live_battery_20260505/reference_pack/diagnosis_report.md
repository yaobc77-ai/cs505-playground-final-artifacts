# Reflection Failure-Mode Diagnosis

Run root: `<PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\reference_pack`
Seeds: 0

## Full Episode Summary

- Success rate: 0.820
- Avg reward: 0.803
- Avg steps: 1.450
- Avg asks: 0.450

## Full vs No Reflection

- Full regretful ASK rate: 0.244
- No reflection regretful ASK rate: n/a
- Full choose regret rate: 0.160
- No reflection choose regret rate: n/a
- Full selected ASK rate: 0.310
- No reflection selected ASK rate: n/a
- Full over-ask cost per step: 0.003
- No reflection over-ask cost per step: n/a
- Full under-ask cost per step: 0.107
- No reflection under-ask cost per step: n/a
- Full choose-without-prior-ask rate: 0.590
- No reflection choose-without-prior-ask rate: n/a

## Tool-Use Summary

- Full scan usage rate: 0.207
- No reflection scan usage rate: n/a
- Full verify usage rate: 0.103
- No reflection verify usage rate: n/a
- Full tool precision: 1.000 (tool_choice_step_count=45)
- No reflection tool precision: n/a (tool_choice_step_count=0)
- Full wrong-tool rate: 0.000
- No reflection wrong-tool rate: n/a
- Full over-tool cost per step: 0.000
- No reflection over-tool cost per step: n/a
- Full under-tool cost per step: 0.000
- No reflection under-tool cost per step: n/a

## Runtime Tool Diagnosis

- Full candidate backend template rate: 0.014
- Full candidate backend union rate: 0.000
- Full oracle tool-choice step count (ask-opportunity proxy): 145.000
- Full oracle family gap mean: 0.306
- Full final oracle family alignment rate: 0.966
- Full final oracle scan coverage rate: 0.759
- Full final oracle verify coverage rate: 0.690
- Full selector family override rate: 0.000
- No reflection oracle tool-choice step count (ask-opportunity proxy): n/a
- No reflection final oracle family alignment rate: n/a
- No reflection selector family override rate: n/a

## Candidate-Layer Diagnosis

### full

- Selected-vs-oracle mismatch rate: 0.690
- Choose-swallow rate: 0.690
- Surface-family distortion rate: 0.000
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.312
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 1.000
- Best-verify missing reasons: not_generated=0.048, generated_but_pruned=0.000, generated_noncanonical=0.000, dominated_after_scoring=0.828
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

### selector_penalty

- Selected-vs-oracle mismatch rate: 0.909
- Choose-swallow rate: 0.909
- Surface-family distortion rate: 0.000
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.336
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 1.000
- Best-verify missing reasons: not_generated=0.000, generated_but_pruned=0.000, generated_noncanonical=0.000, dominated_after_scoring=0.909
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

### voi_memory_v1

- Selected-vs-oracle mismatch rate: 0.529
- Choose-swallow rate: 0.529
- Surface-family distortion rate: 0.000
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.518
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 0.471
- Best-verify missing reasons: not_generated=0.000, generated_but_pruned=0.529, generated_noncanonical=0.000, dominated_after_scoring=0.265
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

## Verify Lineage Survival Audit

### full

- Verify lineage count: 674
- Generated: denominator=674, applicable=674, retained=147, retained_rate=0.218
- Family merge: denominator=674, applicable=147, retained=147, retained_rate=1.000
- Normalization merge: denominator=674, applicable=147, retained=147, retained_rate=1.000
- Dedup: denominator=674, applicable=147, retained=147, retained_rate=1.000
- Candidate cap (source-side): denominator=674, applicable=674, retained=147, retained_rate=0.218
- Pre-selector filter: denominator=674, applicable=154, retained=154, retained_rate=1.000
- Selector entry: denominator=674, applicable=154, retained=154, retained_rate=1.000
- Retained before selector rate: 0.229
- Injected before selector: 0.010
- Shape replacement detected: 0.159
- Pruned stage distribution: generated=0.488, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.283, pre_selector_filter=0.000, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.006, deduplicated=0.000, budget_cap=0.277, filter_rule=0.000, unknown=0.488

### selector_penalty

- Verify lineage count: 540
- Generated: denominator=540, applicable=540, retained=118, retained_rate=0.218
- Family merge: denominator=540, applicable=118, retained=118, retained_rate=1.000
- Normalization merge: denominator=540, applicable=118, retained=118, retained_rate=1.000
- Dedup: denominator=540, applicable=118, retained=118, retained_rate=1.000
- Candidate cap (source-side): denominator=540, applicable=540, retained=118, retained_rate=0.218
- Pre-selector filter: denominator=540, applicable=118, retained=118, retained_rate=1.000
- Selector entry: denominator=540, applicable=118, retained=118, retained_rate=1.000
- Retained before selector rate: 0.218
- Injected before selector: 0.000
- Shape replacement detected: 0.156
- Pruned stage distribution: generated=0.504, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.278, pre_selector_filter=0.000, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.000, deduplicated=0.000, budget_cap=0.278, filter_rule=0.000, unknown=0.504

### voi_memory_v1

- Verify lineage count: 856
- Generated: denominator=856, applicable=856, retained=326, retained_rate=0.381
- Family merge: denominator=856, applicable=326, retained=326, retained_rate=1.000
- Normalization merge: denominator=856, applicable=326, retained=326, retained_rate=1.000
- Dedup: denominator=856, applicable=326, retained=326, retained_rate=1.000
- Candidate cap (source-side): denominator=856, applicable=856, retained=326, retained_rate=0.381
- Pre-selector filter: denominator=856, applicable=326, retained=171, retained_rate=0.524
- Selector entry: denominator=856, applicable=326, retained=171, retained_rate=0.524
- Retained before selector rate: 0.381
- Injected before selector: 0.000
- Shape replacement detected: 0.183
- Pruned stage distribution: generated=0.619, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.000, pre_selector_filter=0.181, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.000, deduplicated=0.000, budget_cap=0.000, filter_rule=0.181, unknown=0.619

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
- `voi_memory_v1` pre-selector summary: denominator=856, applicable_count=326, retained_count=171, retained_rate=0.524
- Filter-rule by ask_gate_reason: strong_negative_voi=155
- Filter-rule by tool_subtype: scan_then_verify=78, verify_first=77
- Verify vs scan pre-selector survival:
  - verify: applicable_count=326, retained_count=171, retained_rate=0.524
  - scan: applicable_count=439, retained_count=193, retained_rate=0.440
- Direct gate evidence coverage: filter_rule_without_direct_gate_evidence=0

## Blocked ASK Value Audit

- Claim boundary: This audit classifies blocked strong_negative_voi steps into task pressure, surfaced ask deficit, or hard cutoff candidates; it does not recommend changing the gate.
- Blocked step count: 100
- task_pressure: count=100, rate=1.000
- surfaced_ask_deficit: count=0, rate=0.000
- hard_cutoff_candidate: count=0, rate=0.000
- Direct choose correct rate: 1.000
- Surface gap distribution: count=100, mean=0.000, median=0.000, positive_rate=0.000
- Suggested next route: task_pressure
- Requires new experiment: false

## Bottleneck Routing

- Routing modes: full, selector_penalty, voi_memory_v1
- Most likely layer: candidate_coverage
- Candidate-related share: 1.000
- Policy-family error share: 0.000

## selector_penalty Snapshot

- Success rate: 0.910
- Avg reward: 0.906
- Avg steps: 1.100
- Avg asks: 0.100
- Regretful ASK rate: 0.000
- Choose regret rate: 0.070
- Over-ask cost per step: 0.000
- Under-ask cost per step: 0.060
- Scan usage rate: 0.054
- Verify usage rate: 0.036
- Tool precision: 1.000 (tool_choice_step_count=10)
- Wrong-tool rate: 0.000
- Over-tool cost per step: 0.000
- Under-tool cost per step: 0.000
- Candidate backend template rate: 0.000
- Candidate backend union rate: 0.000
- Oracle tool-choice step count (ask-opportunity proxy): 110.000
- Oracle family gap mean: 0.375
- Final oracle family alignment rate: 0.864
- Final oracle scan coverage rate: 0.846
- Final oracle verify coverage rate: 0.718
- Selector family override rate: 0.000

## voi_memory_v1 Snapshot

- Success rate: 1.000
- Avg reward: 0.968
- Avg steps: 1.890
- Avg asks: 0.890
- Regretful ASK rate: 0.000
- Choose regret rate: 0.000
- Over-ask cost per step: 0.000
- Under-ask cost per step: 0.000
- Scan usage rate: 0.265
- Verify usage rate: 0.206
- Tool precision: 1.000 (tool_choice_step_count=89)
- Wrong-tool rate: 0.000
- Over-tool cost per step: 0.000
- Under-tool cost per step: 0.000
- Candidate backend template rate: 0.005
- Candidate backend union rate: 0.995
- Oracle tool-choice step count (ask-opportunity proxy): 189.000
- Oracle family gap mean: 0.231
- Final oracle family alignment rate: 1.000
- Final oracle scan coverage rate: 0.852
- Final oracle verify coverage rate: 0.677
- Selector family override rate: 0.000
- Ask-gate block rate: 0.529
- Ask-gate allow rate given positive ask advantage: 1.000
- Ask-gate block rate given nonpositive ask advantage: 1.000
- Retrieved positive evidence rate: 0.995
- Retrieved caution evidence rate: 0.989

## Difficulty Breakdown

| Group | Full regretful ASK | No reflection regretful ASK | Full choose regret | No reflection choose regret | Full over-ask cost | Full under-ask cost |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 0.150 | n/a | 0.188 | n/a | 0.001 | 0.138 |
| medium | 0.320 | n/a | 0.111 | n/a | 0.007 | 0.063 |

## Noise Breakdown

| Group | Full regretful ASK | No reflection regretful ASK | Full choose regret | No reflection choose regret | Full over-ask cost | Full under-ask cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.150 | n/a | 0.188 | n/a | 0.001 | 0.138 |
| 0.18 | 0.320 | n/a | 0.111 | n/a | 0.007 | 0.063 |

## Reflection Avoid-Action Distribution

- avoid_actions=['CHOOSE']: 0.580 (58)
- avoid_actions=['ASK']: 0.350 (35)
- avoid_actions=[]: 0.060 (6)
- mixed avoid actions: 0.010 (1)

## Reflection Delta Summary

- info_seeking: mean=0.142, abs_mean=0.256, positive=0.600, negative=0.320
- cost_sensitivity: mean=0.046, abs_mean=0.106, positive=0.380, negative=0.180
- confidence_threshold: mean=0.205, abs_mean=0.246, positive=0.670, negative=0.190

## Subsequent Correlation Summary

- avoid CHOOSE -> next ask count: -0.128
- avoid CHOOSE -> next reward: -0.072
- confidence threshold delta -> next ask count: -0.083
- info seeking delta -> next ask count: -0.104

## Memory-Mediated CHOOSE Suppression

- Selected ASK with retrieved CHOOSE-suppressing memory: 0.977
- Regretful ASK with CHOOSE-suppressing memory: 1.000

## Under-asking Profile

- Full choose regret rate: 0.160
- Full under-ask cost per choose regret: 0.966
- Full choose regret without prior ask rate: 1.000
- Full choose regret high-margin share: 0.000

## Recommendation

- Recommended next intervention: `voi_memory_v1`
- voi_memory_v1 has the strongest experimental profile: avg_reward=0.968, success_rate=1.000.
- voi_memory_v1 keeps regretful_ask_rate=0.000 and choose_regret_rate=0.000.
- voi_memory_v1 keeps over_ask_cost_per_step=0.000 and under_ask_cost_per_step=0.000.
- voi_memory_v1 keeps tool_precision=1.000, wrong_tool_rate=0.000, over_tool_cost_per_step=0.000, under_tool_cost_per_step=0.000.
- Compared with full: reward delta=0.165, success delta=0.180.
