# Pruning-Path Audit

Run root: `<PLAYGROUND_ROOT>\logs\v4_seed7_live_battery_20260505\reference_pack`
Seeds: 0

## Focus

- Current focus: `pruning_path_audit`
- Execution lane: `discovery`
- Scope: verify lineage survival only; no memory, gate, selector, task-generator, or benchmark expansion changes are interpreted here.

## full

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

## selector_penalty

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

## voi_memory_v1

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

## Routing

- Most likely layer: candidate_coverage
- Candidate-related share: 1.000
- Policy-family error share: 0.000
