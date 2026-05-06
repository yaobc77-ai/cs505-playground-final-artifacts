# Pre-Selector Pruning Audit

Run root: `<PLAYGROUND_ROOT>\logs\seed11_v3_same_design_collapse_probe_20260423\stress_pack`
Seeds: 0

## Focus

- Current focus: `pre_selector_pruning_audit`
- Execution lane: `discovery`
- Claim boundary: This audit attributes which gate-facing reason accounted for the block; it does not recommend changing the gate.
- Scope: attribution only for gate-facing block reasons on the existing live seed-7 run; no gate-change conclusion is allowed here.

## Evidence Schema

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

## full

- Claim boundary: block attribution only; this audit does not recommend changing the gate.
- Verify pre-selector summary: denominator=869, applicable_count=165, retained_count=165, retained_rate=1.000
- Filter-rule count: 0
- Direct gate evidence coverage: filter_rule_without_direct_gate_evidence=0
- Filter-rule by ask_gate_reason: none
- Filter-rule by tool_subtype: none
- Verify vs scan pre-selector survival:
  - verify: applicable_count=165, retained_count=165, retained_rate=1.000
  - scan: applicable_count=220, retained_count=220, retained_rate=1.000
- Secondary context comparison (descriptive only):
- ask_gate_score: filter_rule_mean=n/a, entered_selector_mean=n/a
- ask_advantage: filter_rule_mean=n/a, entered_selector_mean=0.276
- best_ask_family: filter_rule_mean=n/a, entered_selector_mean=n/a
- best_ask_eval: filter_rule_mean=n/a, entered_selector_mean=n/a
- best_choose_eval: filter_rule_mean=n/a, entered_selector_mean=n/a
- voi_ratio: filter_rule_mean=n/a, entered_selector_mean=n/a
- uncertainty: filter_rule_mean=n/a, entered_selector_mean=n/a
- base_uncertainty: filter_rule_mean=n/a, entered_selector_mean=n/a
- reliability_uncertainty: filter_rule_mean=n/a, entered_selector_mean=n/a
- evidence_conflict: filter_rule_mean=n/a, entered_selector_mean=n/a
- memory_signal: filter_rule_mean=n/a, entered_selector_mean=n/a
- helpful_ask: filter_rule_mean=n/a, entered_selector_mean=n/a
- choose_regret: filter_rule_mean=n/a, entered_selector_mean=n/a
- regretful_ask: filter_rule_mean=n/a, entered_selector_mean=n/a
- choose_helpful: filter_rule_mean=n/a, entered_selector_mean=n/a

## selector_penalty

- Claim boundary: block attribution only; this audit does not recommend changing the gate.
- Verify pre-selector summary: denominator=745, applicable_count=134, retained_count=134, retained_rate=1.000
- Filter-rule count: 0
- Direct gate evidence coverage: filter_rule_without_direct_gate_evidence=0
- Filter-rule by ask_gate_reason: none
- Filter-rule by tool_subtype: none
- Verify vs scan pre-selector survival:
  - verify: applicable_count=134, retained_count=134, retained_rate=1.000
  - scan: applicable_count=182, retained_count=182, retained_rate=1.000
- Secondary context comparison (descriptive only):
- ask_gate_score: filter_rule_mean=n/a, entered_selector_mean=n/a
- ask_advantage: filter_rule_mean=n/a, entered_selector_mean=0.219
- best_ask_family: filter_rule_mean=n/a, entered_selector_mean=n/a
- best_ask_eval: filter_rule_mean=n/a, entered_selector_mean=n/a
- best_choose_eval: filter_rule_mean=n/a, entered_selector_mean=n/a
- voi_ratio: filter_rule_mean=n/a, entered_selector_mean=n/a
- uncertainty: filter_rule_mean=n/a, entered_selector_mean=n/a
- base_uncertainty: filter_rule_mean=n/a, entered_selector_mean=n/a
- reliability_uncertainty: filter_rule_mean=n/a, entered_selector_mean=n/a
- evidence_conflict: filter_rule_mean=n/a, entered_selector_mean=n/a
- memory_signal: filter_rule_mean=n/a, entered_selector_mean=n/a
- helpful_ask: filter_rule_mean=n/a, entered_selector_mean=n/a
- choose_regret: filter_rule_mean=n/a, entered_selector_mean=n/a
- regretful_ask: filter_rule_mean=n/a, entered_selector_mean=n/a
- choose_helpful: filter_rule_mean=n/a, entered_selector_mean=n/a

## voi_memory_v1

- Claim boundary: block attribution only; this audit does not recommend changing the gate.
- Verify pre-selector summary: denominator=1015, applicable_count=364, retained_count=169, retained_rate=0.464
- Filter-rule count: 195
- Direct gate evidence coverage: filter_rule_without_direct_gate_evidence=0
- Filter-rule by ask_gate_reason: strong_negative_voi=195
- Filter-rule by tool_subtype: deceptive_scan=105, scan_then_verify=51, verify_first=39
- Verify vs scan pre-selector survival:
  - verify: applicable_count=364, retained_count=169, retained_rate=0.464
  - scan: applicable_count=413, retained_count=206, retained_rate=0.499
- Secondary context comparison (descriptive only):
- ask_gate_score: filter_rule_mean=n/a, entered_selector_mean=n/a
- ask_advantage: filter_rule_mean=-0.069, entered_selector_mean=0.960
- best_ask_family: filter_rule_mean=n/a, entered_selector_mean=n/a
- best_ask_eval: filter_rule_mean=0.931, entered_selector_mean=0.960
- best_choose_eval: filter_rule_mean=1.000, entered_selector_mean=0.000
- voi_ratio: filter_rule_mean=-1.440, entered_selector_mean=22.044
- uncertainty: filter_rule_mean=0.204, entered_selector_mean=1.000
- base_uncertainty: filter_rule_mean=0.169, entered_selector_mean=1.000
- reliability_uncertainty: filter_rule_mean=0.307, entered_selector_mean=1.000
- evidence_conflict: filter_rule_mean=0.000, entered_selector_mean=0.000
- memory_signal: filter_rule_mean=0.010, entered_selector_mean=0.000
- helpful_ask: filter_rule_mean=1.000, entered_selector_mean=0.988
- choose_regret: filter_rule_mean=0.000, entered_selector_mean=0.000
- regretful_ask: filter_rule_mean=0.000, entered_selector_mean=0.000
- choose_helpful: filter_rule_mean=0.990, entered_selector_mean=0.988