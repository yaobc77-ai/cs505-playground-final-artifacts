# Pruning-Path Audit

Run root: `<PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\stress_pack`
Seeds: 0

## Focus

- Current focus: `pruning_path_audit`
- Execution lane: `discovery`
- Scope: verify lineage survival only; no memory, gate, selector, task-generator, or benchmark expansion changes are interpreted here.

## full

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

## selector_penalty

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

## voi_memory_v1

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

## Routing

- Most likely layer: candidate_coverage
- Candidate-related share: 1.000
- Policy-family error share: 0.000
