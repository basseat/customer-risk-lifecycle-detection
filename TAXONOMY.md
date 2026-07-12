# Event Taxonomy: Customer Risk Lifecycle Tracking Design
### Based on the German/EU AML risk classification framework (GwG, BaFin, EBA)

**Purpose:** define what needs to be captured across a customer's risk journey so structuring, pass-through, and mule-network patterns can actually be detected, not just described after the fact. This is the design document behind the schema used in [`generate_data.py`](./generate_data.py) and [`detection_queries.sql`](./detection_queries.sql).

---

## The framework

Germany's AML law is the **Geldwäschegesetz (GwG)**, supervised by **BaFin**, implementing the EU's AML Directives and increasingly the EU AML Regulation (AMLR). Risk is classified into three categories that can apply to any customer, and can change over time based on behaviour, not just at account opening:

| Category | German/EU term | When it applies | Due diligence level |
|---|---|---|---|
| **Lower risk** | Vereinfachte Sorgfaltspflichten (SDD) | Customer, product, or geography factors from GwG Annex 1 indicate low ML/TF risk (e.g. public companies with strong disclosure, low-risk jurisdictions, financial-inclusion products) | Simplified Due Diligence, reduced verification, update cycle: risk-appropriate, no fixed maximum |
| **Normal risk** | Allgemeine Sorgfaltspflichten (CDD) | The default category, no lower- or higher-risk factors present | Standard Customer Due Diligence, update cycle: at least every 5 years |
| **Higher risk** | Verstärkte Sorgfaltspflichten (EDD) | GwG Annex 2 factors present: PEPs, high-risk jurisdictions, complex/opaque structures, unusual transaction patterns, high-risk products | Enhanced Due Diligence: senior management approval, source-of-funds verification, ongoing enhanced monitoring, update cycle: at least every 1 year |

A few features of this framework that shaped the schema design:

- **It's risk-based, not fixed at onboarding.** A customer isn't locked into a category at signup; classification reflects assessed risk and can be revisited.
- **PEP status specifically triggers EDD**, and stays in force for at least 12 months after the person leaves the political role.
- **Filing a Suspicious Activity Report (SAR) itself changes classification.** Under BaFin's 2025 guidance, filing a SAR automatically puts the customer into a heightened-risk category for **21 calendar days** (money laundering suspicion) or **6 months** (terrorist financing suspicion), even before the FIU responds.
- **Update cycles are explicit**: normal risk, at least every 5 years; higher risk, at least every 1 year (tightened from 2 years); lower risk, no fixed maximum but must be risk-appropriate.
- **Germany's FIU (Zentralstelle für Finanztransaktionsuntersuchungen)** sits within the Zollkriminalamt (Customs Investigation Bureau), separate from BaFin. Reports go to the FIU; supervision and enforcement sit with BaFin.

---

## 1. `account_opened`
Captured once, at onboarding.
- `account_id`
- `initial_risk_category` (lower_risk_sdd, normal_risk_cdd, higher_risk_edd)
- `risk_factors_present` (list, referencing GwG Annex 1 lower-risk or Annex 2 higher-risk factors, e.g. `low_risk_jurisdiction`, `financial_inclusion_product`, `pep_status`, `complex_structure`)
- `kyc_documentation_completeness` (0 to 1 score, or flag list of missing fields)
- `onboarding_channel` (app, branch, partner_institution)
- `onboarding_device_id`
- `opened_at`

**Why it matters:** `initial_risk_category` must be traceable back to the specific Annex 1 or Annex 2 factor that justified it, since BaFin expects a documented, factor-based risk analysis, not just a label.

---

## 2. `transaction_event`
Captured on every transaction.
- `transaction_id`
- `account_id`
- `direction` (inflow, outflow)
- `amount`
- `counterparty_id`
- `counterparty_country`
- `channel` (transfer, card, ATM, agent)
- `timestamp`

**Why it matters:** `counterparty_country` matters specifically under the German/EU framework, since geography is an explicit Annex 1/Annex 2 risk factor (low-risk vs high-risk third countries), not just a nice-to-have field.

---

## 3. `velocity_flag_triggered`
Captured whenever a rule-based or statistical threshold fires.
- `account_id`
- `flag_type` (high_frequency_low_value, rapid_pass_through, threshold_breach)
- `window` (e.g. rolling_7_day)
- `threshold_breached` (the specific rule and value, e.g. the €1,000 transfer-information threshold or the €15,000 general CDD trigger under GwG)
- `triggered_at`

**Why it matters:** German law has explicit statutory thresholds (€1,000 for transfer-of-funds information requirements, €15,000 for general transaction-triggered CDD), so `threshold_breached` should reference the actual regulatory trigger where relevant, not just an internal arbitrary number.

---

## 4. `case_escalated`
Captured when a flag moves to human investigation.
- `case_id`
- `account_id`
- `escalation_reason`
- `risk_category_at_time` (may differ from category at opening)
- `escalation_path` (auto_to_analyst, analyst_to_senior_management, cross_account_cluster)
- `senior_management_approval` (boolean, required under GwG for higher-risk relationships)
- `escalated_at`

**Why it matters:** `senior_management_approval` isn't optional detail, it's a specific German legal requirement for establishing or continuing a higher-risk business relationship, so it needs to exist as a trackable field, not just a process step done off-system.

---

## 5. `network_link_identified`
Captured when accounts are linked through shared attributes.
- `account_id_a`, `account_id_b`
- `link_type` (shared_beneficiary, shared_address, shared_device, shared_identifying_document)
- `identified_at`

**Why it matters:** this is the field that turns "one odd account" into "a coordinated network." In this project's implementation, the strongest of these signals is shared onboarding device, multiple identities completing remote video-identification from the same device is a well-documented mule-recruitment indicator in EU digital banking.

---

## 6. `case_resolved`
Captured when an investigation closes.
- `case_id`
- `outcome` (cleared, restricted, offboarded, sar_filed)
- `sar_filed` (boolean)
- `sar_filed_at`
- `heightened_risk_window_end` (derived: sar_filed_at + 21 days for ML suspicion, or +6 months for terrorist-financing suspicion, per current BaFin guidance)
- `time_to_resolution` (derived: resolved_at minus escalated_at)
- `resolved_at`

**Why it matters:** `heightened_risk_window_end` reflects a real, current, specific German rule, a SAR filing automatically imposes a defined heightened-risk period regardless of FIU response time, so the system needs to track that window explicitly, not leave it to memory.

---

## 7. `risk_re_rated`
Captured whenever an account's risk category changes.
- `account_id`
- `old_category`, `new_category`
- `trigger_reason` (case_outcome, periodic_review, sar_filing, pep_status_change, behaviour_change)
- `next_review_due` (derived from category: lower_risk = risk-appropriate/no fixed max, normal_risk = +5 years, higher_risk = +1 year)
- `re_rated_at`

**Why it matters:** the review cadence is a specific, current regulatory number per category (5 years normal, 1 year higher-risk, tightened from 2 years under the 2025 BaFin guidance), so `next_review_due` should be a derived, trackable field, not an ad hoc calendar reminder.

---

## Design principle

Structuring is only visible in aggregate, so transaction-level detail has to support rolling-window aggregation. Under the German/EU framework specifically, risk category, review cadence, and escalation approval are all governed by specific statutory rules (GwG Annexes 1 and 2, BaFin's 2025 guidance), so the taxonomy has to make those rules traceable and auditable, not just internally consistent.

## Implementation

`account_opened` and `network_link_identified` (device signal) are implemented in [`generate_data.py`](./generate_data.py) and detected via [`detection_queries.sql`](./detection_queries.sql). The remaining event types (`velocity_flag_triggered`, `case_escalated`, `case_resolved`, `risk_re_rated`) describe the fuller lifecycle this schema is designed for; the current implementation focuses on the detection layer (events 1, 2, 3, 5) as a first working slice.
