# Pruning-Path Audit

Run root: `<PLAYGROUND_ROOT>\logs\seed11_v3_same_design_collapse_probe_20260423\stress_pack`
Seeds: 0

## Focus

- Current focus: `pruning_path_audit`
- Execution lane: `discovery`
- Scope: verify lineage survival only; no memory, gate, selector, task-generator, or benchmark expansion changes are interpreted here.

## full

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

## selector_penalty

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

## voi_memory_v1

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

## Routing

- Most likely layer: candidate_coverage
- Candidate-related share: 1.000
- Policy-family error share: 0.000
