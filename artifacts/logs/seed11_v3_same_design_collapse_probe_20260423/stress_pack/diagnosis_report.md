# Reflection Failure-Mode Diagnosis

Run root: `<PLAYGROUND_ROOT>\logs\seed11_v3_same_design_collapse_probe_20260423\stress_pack`
Seeds: 0

## Full Episode Summary

- Success rate: 0.740
- Avg reward: 0.697
- Avg steps: 1.570
- Avg asks: 0.570

## Full vs No Reflection

- Full regretful ASK rate: 0.298
- No reflection regretful ASK rate: n/a
- Full choose regret rate: 0.151
- No reflection choose regret rate: n/a
- Full selected ASK rate: 0.363
- No reflection selected ASK rate: n/a
- Full over-ask cost per step: 0.008
- No reflection over-ask cost per step: n/a
- Full under-ask cost per step: 0.088
- No reflection under-ask cost per step: n/a
- Full choose-without-prior-ask rate: 0.490
- No reflection choose-without-prior-ask rate: n/a

## Tool-Use Summary

- Full scan usage rate: 0.057
- No reflection scan usage rate: n/a
- Full verify usage rate: 0.306
- No reflection verify usage rate: n/a
- Full tool precision: 1.000 (tool_choice_step_count=57)
- No reflection tool precision: n/a (tool_choice_step_count=0)
- Full wrong-tool rate: 0.000
- No reflection wrong-tool rate: n/a
- Full over-tool cost per step: 0.000
- No reflection over-tool cost per step: n/a
- Full under-tool cost per step: 0.000
- No reflection under-tool cost per step: n/a

## Runtime Tool Diagnosis

- Full candidate backend template rate: 0.038
- Full candidate backend union rate: 0.000
- Full oracle tool-choice step count (ask-opportunity proxy): 156.000
- Full oracle family gap mean: 0.220
- Full final oracle family alignment rate: 0.609
- Full final oracle scan coverage rate: 0.083
- Full final oracle verify coverage rate: 0.494
- Full selector family override rate: 0.000
- No reflection oracle tool-choice step count (ask-opportunity proxy): n/a
- No reflection final oracle family alignment rate: n/a
- No reflection selector family override rate: n/a

## Candidate-Layer Diagnosis

### full

- Selected-vs-oracle mismatch rate: 0.840
- Choose-swallow rate: 0.635
- Surface-family distortion rate: 0.205
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.348
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 1.000
- Best-verify missing reasons: not_generated=0.115, generated_but_pruned=0.000, generated_noncanonical=0.000, dominated_after_scoring=0.564
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

### selector_penalty

- Selected-vs-oracle mismatch rate: 0.922
- Choose-swallow rate: 0.775
- Surface-family distortion rate: 0.147
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.325
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 1.000
- Best-verify missing reasons: not_generated=0.116, generated_but_pruned=0.000, generated_noncanonical=0.000, dominated_after_scoring=0.574
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

### voi_memory_v1

- Selected-vs-oracle mismatch rate: 0.546
- Choose-swallow rate: 0.546
- Surface-family distortion rate: 0.000
- Policy-family error rate: 0.000
- Best-verify canonical surface before: 0.361
- Best-verify canonical surface after: 1.000
- Verify candidate exposed rate: 0.454
- Best-verify missing reasons: not_generated=0.000, generated_but_pruned=0.546, generated_noncanonical=0.000, dominated_after_scoring=0.355
- Best-scan surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000
- Best-verify surface-shape distribution: canonical=1.000, fixed=0.000, ask_only=0.000, multi_step=0.000

## Verify Lineage Survival Audit

### full

- Verify lineage count: 869
- Generated: denominator=869, applicable=869, retained=147, retained_rate=0.169
- Family merge: denominator=869, applicable=147, retained=147, retained_rate=1.000
- Normalization merge: denominator=869, applicable=147, retained=147, retained_rate=1.000
- Dedup: denominator=869, applicable=147, retained=147, retained_rate=1.000
- Candidate cap (source-side): denominator=869, applicable=869, retained=147, retained_rate=0.169
- Pre-selector filter: denominator=869, applicable=165, retained=165, retained_rate=1.000
- Selector entry: denominator=869, applicable=165, retained=165, retained_rate=1.000
- Retained before selector rate: 0.190
- Injected before selector: 0.021
- Shape replacement detected: 0.114
- Pruned stage distribution: generated=0.539, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.272, pre_selector_filter=0.000, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.011, deduplicated=0.000, budget_cap=0.260, filter_rule=0.000, unknown=0.539

### selector_penalty

- Verify lineage count: 745
- Generated: denominator=745, applicable=745, retained=119, retained_rate=0.160
- Family merge: denominator=745, applicable=119, retained=119, retained_rate=1.000
- Normalization merge: denominator=745, applicable=119, retained=119, retained_rate=1.000
- Dedup: denominator=745, applicable=119, retained=119, retained_rate=1.000
- Candidate cap (source-side): denominator=745, applicable=745, retained=119, retained_rate=0.160
- Pre-selector filter: denominator=745, applicable=134, retained=134, retained_rate=1.000
- Selector entry: denominator=745, applicable=134, retained=134, retained_rate=1.000
- Retained before selector rate: 0.180
- Injected before selector: 0.020
- Shape replacement detected: 0.111
- Pruned stage distribution: generated=0.553, family_merge=0.000, normalization_merge=0.000, dedup=0.000, candidate_cap=0.267, pre_selector_filter=0.000, unknown=0.000
- Pruned reason distribution: shape_replaced_by_equivalent=0.000, dominated_within_family=0.000, dominated_cross_family=0.020, deduplicated=0.000, budget_cap=0.247, filter_rule=0.000, unknown=0.553

### voi_memory_v1

- Verify lineage count: 1015
- Generated: denominator=1015, applicable=1015, retained=364, retained_rate=0.359
- Family merge: denominator=1015, applicable=364, retained=364, retained_rate=1.000
- Normalization merge: denominator=1015, applicable=364, retained=364, retained_rate=1.000
- Dedup: denominator=1015, applicable=364, retained=364, retained_rate=1.000
- Candidate cap (source-side): denominator=1015, applicable=1015, retained=364, retained_rate=0.359
- Pre-selector filter: denominator=1015, applicable=364, retained=169, retained_rate=0.464
- Selector entry: denominator=1015, applicable=364, retained=169, retained_rate=0.464
- Retained before selector rate: 0.359
- Injected before selector: 0.000
- Shape replacement detected: 0.153
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
- `voi_memory_v1` pre-selector summary: denominator=1015, applicable_count=364, retained_count=169, retained_rate=0.464
- Filter-rule by ask_gate_reason: strong_negative_voi=195
- Filter-rule by tool_subtype: deceptive_scan=105, scan_then_verify=51, verify_first=39
- Verify vs scan pre-selector survival:
  - verify: applicable_count=364, retained_count=169, retained_rate=0.464
  - scan: applicable_count=413, retained_count=206, retained_rate=0.499
- Direct gate evidence coverage: filter_rule_without_direct_gate_evidence=0

## Blocked ASK Value Audit

- Claim boundary: This audit classifies blocked strong_negative_voi steps into task pressure, surfaced ask deficit, or hard cutoff candidates; it does not recommend changing the gate.
- Blocked step count: 100
- task_pressure: count=100, rate=1.000
- surfaced_ask_deficit: count=0, rate=0.000
- hard_cutoff_candidate: count=0, rate=0.000
- Direct choose correct rate: 1.000
- Surface gap distribution: count=100, mean=0.036, median=0.030, positive_rate=0.650
- Suggested next route: task_pressure
- Requires new experiment: false

## Bottleneck Routing

- Routing modes: full, selector_penalty, voi_memory_v1
- Most likely layer: candidate_coverage
- Candidate-related share: 1.000
- Policy-family error share: 0.000

## selector_penalty Snapshot

- Success rate: 0.750
- Avg reward: 0.727
- Avg steps: 1.290
- Avg asks: 0.290
- Regretful ASK rate: 0.138
- Choose regret rate: 0.160
- Over-ask cost per step: 0.001
- Under-ask cost per step: 0.113
- Scan usage rate: 0.031
- Verify usage rate: 0.194
- Tool precision: 1.000 (tool_choice_step_count=29)
- Wrong-tool rate: 0.000
- Over-tool cost per step: 0.000
- Under-tool cost per step: 0.000
- Candidate backend template rate: 0.008
- Candidate backend union rate: 0.000
- Oracle tool-choice step count (ask-opportunity proxy): 129.000
- Oracle family gap mean: 0.248
- Final oracle family alignment rate: 0.488
- Final oracle scan coverage rate: 0.062
- Final oracle verify coverage rate: 0.597
- Selector family override rate: 0.000

## voi_memory_v1 Snapshot

- Success rate: 1.000
- Avg reward: 0.964
- Avg steps: 1.830
- Avg asks: 0.830
- Regretful ASK rate: 0.000
- Choose regret rate: 0.000
- Over-ask cost per step: 0.000
- Under-ask cost per step: 0.000
- Scan usage rate: 0.355
- Verify usage rate: 0.098
- Tool precision: 1.000 (tool_choice_step_count=83)
- Wrong-tool rate: 0.000
- Over-tool cost per step: 0.000
- Under-tool cost per step: 0.000
- Candidate backend template rate: 0.000
- Candidate backend union rate: 1.000
- Oracle tool-choice step count (ask-opportunity proxy): 183.000
- Oracle family gap mean: 0.177
- Final oracle family alignment rate: 0.645
- Final oracle scan coverage rate: 0.421
- Final oracle verify coverage rate: 0.836
- Selector family override rate: 0.000
- Ask-gate block rate: 0.546
- Ask-gate allow rate given positive ask advantage: 1.000
- Ask-gate block rate given nonpositive ask advantage: 1.000
- Retrieved positive evidence rate: 0.995
- Retrieved caution evidence rate: 0.989

## Difficulty Breakdown

| Group | Full regretful ASK | No reflection regretful ASK | Full choose regret | No reflection choose regret | Full over-ask cost | Full under-ask cost |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 0.136 | n/a | 0.054 | n/a | 0.003 | 0.032 |
| hard | 0.500 | n/a | 0.176 | n/a | 0.025 | 0.096 |
| medium | 0.348 | n/a | 0.222 | n/a | 0.004 | 0.132 |

## Noise Breakdown

| Group | Full regretful ASK | No reflection regretful ASK | Full choose regret | No reflection choose regret | Full over-ask cost | Full under-ask cost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.136 | n/a | 0.054 | n/a | 0.003 | 0.032 |
| 0.18 | 0.348 | n/a | 0.222 | n/a | 0.004 | 0.132 |
| 0.32 | 0.500 | n/a | 0.176 | n/a | 0.025 | 0.096 |

## Reflection Avoid-Action Distribution

- avoid_actions=['CHOOSE']: 0.480 (48)
- avoid_actions=['ASK']: 0.430 (43)
- avoid_actions=[]: 0.050 (5)
- mixed avoid actions: 0.040 (4)

## Reflection Delta Summary

- info_seeking: mean=0.114, abs_mean=0.285, positive=0.490, negative=0.420
- cost_sensitivity: mean=0.103, abs_mean=0.147, positive=0.500, negative=0.120
- confidence_threshold: mean=0.182, abs_mean=0.231, positive=0.620, negative=0.220

## Subsequent Correlation Summary

- avoid CHOOSE -> next ask count: 0.152
- avoid CHOOSE -> next reward: 0.254
- confidence threshold delta -> next ask count: 0.183
- info seeking delta -> next ask count: 0.250

## Memory-Mediated CHOOSE Suppression

- Selected ASK with retrieved CHOOSE-suppressing memory: 1.000
- Regretful ASK with CHOOSE-suppressing memory: 1.000

## Under-asking Profile

- Full choose regret rate: 0.151
- Full under-ask cost per choose regret: 0.919
- Full choose regret without prior ask rate: 1.000
- Full choose regret high-margin share: 0.000

## Recommendation

- Recommended next intervention: `voi_memory_v1`
- voi_memory_v1 has the strongest experimental profile: avg_reward=0.964, success_rate=1.000.
- voi_memory_v1 keeps regretful_ask_rate=0.000 and choose_regret_rate=0.000.
- voi_memory_v1 keeps over_ask_cost_per_step=0.000 and under_ask_cost_per_step=0.000.
- voi_memory_v1 keeps tool_precision=1.000, wrong_tool_rate=0.000, over_tool_cost_per_step=0.000, under_tool_cost_per_step=0.000.
- Compared with full: reward delta=0.267, success delta=0.260.
