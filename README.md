# Customer Risk Lifecycle: Structuring & Mule-Network Detection

A synthetic-data pipeline that detects two well-documented financial-crime typologies, **structuring** (breaking transactions into small amounts to stay under reporting thresholds) and **money mule networks** (coordinated accounts recruited to move funds), against the actual German/EU AML risk-classification framework.

Built as applied preparation for a Customer Risk Lifecycle / Anti-Financial Crime analytics role, and as a genuine attempt to think through what that work looks like day to day: from risk classification, to detection logic, to validating a new rule before it ships.

## Why this project exists

Most portfolio projects analyse data that already exists. This one is different: it's built to answer a specific question, **how would you design tracking and detection for a customer's risk lifecycle at a bank operating under German/EU AML law**, and then actually test whether the resulting SQL can find a known bad pattern hidden inside realistic transaction data.

## The regulatory grounding

Account risk categories follow the real German **GwG (Geldwäschegesetz)** framework, supervised by BaFin:

- **Lower risk (SDD)**: Simplified Due Diligence, driven by GwG Annex 1 factors
- **Normal risk (CDD)**: standard due diligence, the default category
- **Higher risk (EDD)**: Enhanced Due Diligence, driven by GwG Annex 2 factors (PEPs, high-risk jurisdictions, complex structures)

The structuring detection threshold (€700) is deliberately set below the real **€1,000 GwG transfer-of-funds information threshold**, the actual statutory trigger a structuring scheme would be designed to avoid.

The network-link detection is built around **shared onboarding device**, a well-documented real-world signal at digital-first banks running remote video-identification (IDnow/WebID-style), where multiple "unrelated" identities completing onboarding from the same device is a recognised mule-recruitment indicator, rather than a signal borrowed from a different market's KYC conventions.

## What's in this repo

| File | What it does |
|---|---|
| [`TAXONOMY.md`](./TAXONOMY.md) | The design document: the full event schema this pipeline is built against, and the German/EU risk classification framework (GwG, BaFin) behind it |
| `generate_data.py` | Generates 5,000 synthetic accounts and ~420k transactions, with a deliberately seeded 18-account structuring/mule cluster hidden inside the lower-risk population, used as ground truth to validate detection |
| `add_near_miss_cases.py` | Adds a second, harder 15-account cluster with noisier, less obvious structuring behaviour (wider amount variance, slower pass-through, weaker device overlap), plus a jurisdiction-risk feature, so rules-based and ML detection can be compared on a genuinely differentiated test rather than an easy one |
| `build_features.py` | Converts the same signals the SQL pipeline detects (velocity, pass-through, network links, jurisdiction risk) into a per-account feature table for machine learning |
| `ml_detection.py` | Supervised (Random Forest, class-weighted) and unsupervised (Isolation Forest) detection, compared against each other and against the rules-based baseline |
| `false_positive_analysis.py` | Examines which non-positive accounts the model ranks highest and why, feature importances, and an honest read of how large the true separation actually is |
| `model_governance.py` | Whether results can be re-run and verified later (seed logging), whether performance is tracked over time rather than known only at a single point (a versioned model registry log), and whether review cadence actually matches the GwG risk-review cycles the underlying categories carry |
| `detection_queries.sql` | Four PostgreSQL views: rolling 7-day inflow velocity (window functions), time-to-outflow (self-join), shared-counterparty and shared-device network links (self-joins), and a combined weighted structuring-likelihood score |
| `run_detection.py` | Loads the queries, runs them against PostgreSQL, and checks recall/precision against the seeded ground truth |
| `setup_database.sh` | One-shot script to create the database, tables, and load the generated CSVs |

## Review cadence

`model_governance.py` computes the next review-due date on an annual cadence. The reasoning: GwG mandates at-minimum-annual review for higher-risk accounts, and a model influencing risk classification should be reviewed at least as often as the riskiest category it affects. A SAR filing also triggers a 21-day (money laundering) or 6-month (terrorist financing) heightened-risk window under current BaFin guidance, which is why every run logs its seeds and feature set, so results are reproducible if a model's output is ever reviewed after the fact.

## Results

**Rules-based SQL detection**, against the original, more obvious 18-account cluster: **100% recall, 90% precision** in the top-20.

**Against a harder, combined test** (the original 18-account cluster plus a second, deliberately noisier 15-account near-miss cluster designed to fall outside fixed thresholds), the three approaches diverge in a genuinely informative way:

| Method | Overall recall | Obvious cluster | Near-miss cluster |
|---|---|---|---|
| Rules-based SQL | 55% | 100% | 0% |
| Supervised Random Forest | 100% | 100% | 100% |
| Unsupervised Isolation Forest | 91% | 100% | 80% |

The takeaway: rules-based detection is excellent at known patterns and blind to anything that falls outside a fixed threshold. Supervised ML, trained on labels, generalises past that limitation. Unsupervised anomaly detection, which never sees a single label, still catches most of the harder cases purely from behavioural deviation, the more realistic scenario for catching a typology nobody has confirmed yet.

A false-positive analysis (`false_positive_analysis.py`) found the gap between true positives and the next-highest-ranked accounts is large on this synthetic dataset, an honest limitation: synthetic data this clean validates that detection logic *works*, but can't validate how any method behaves at a genuinely ambiguous decision boundary, which only real historical alert data with real investigator dispositions can do. Feature importance analysis is still useful regardless: device and pass-through-speed signals dominate over raw transaction volume.

## Interactive dashboard

Published to Tableau Public: [Rules-Based vs. ML Detection: Recall Comparison](https://public.tableau.com/app/profile/abdulbasit.ayoade/viz/CustomerRiskLifecycle/RecallComparison)

Two dashboards: the headline recall comparison (interactive, click a method to drill into its obvious vs. near-miss breakdown), and a second covering what drives the model and where flagged accounts concentrate by risk category.

## How to run it

Requires PostgreSQL and Python 3.

```bash
pip install -r requirements.txt
python generate_data.py          # generates accounts.csv, transactions.csv, seeded_cluster_ground_truth.csv
python add_near_miss_cases.py    # adds the harder near-miss cluster and jurisdiction risk
./setup_database.sh              # creates the database and loads the CSVs
python run_detection.py          # rules-based SQL detection, prints ranked results and recall/precision
python build_features.py         # builds the ML feature table
python ml_detection.py           # supervised + unsupervised detection, compared to the rules-based baseline
python false_positive_analysis.py  # examines the model's highest-ranked non-positive accounts
python model_governance.py       # logs a versioned registry entry, computes the next review-due date
```

## A note on the synthetic data

The risk-category distribution and its correlation with KYC completeness are illustrative modelling choices, not sourced from any public statistic. They exist to test the detection logic, not to model real demographics. The regulatory thresholds and categories themselves are real (GwG Annexes 1 and 2, BaFin interpretation guidance).

## Related project

This pipeline's risk-segmentation logic also extends an existing project: [ab-test-checkout-fraud](https://github.com/basseat/ab-test-checkout-fraud), which tests whether a checkout-fraud A/B test's effect differs significantly by risk category.
