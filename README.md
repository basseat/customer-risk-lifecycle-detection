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
| `generate_data.py` | Generates 5,000 synthetic accounts and ~420k transactions, with a deliberately seeded 18-account structuring/mule cluster hidden inside the lower-risk population, used as ground truth to validate detection |
| `detection_queries.sql` | Four PostgreSQL views: rolling 7-day inflow velocity (window functions), time-to-outflow (self-join), shared-counterparty and shared-device network links (self-joins), and a combined weighted structuring-likelihood score |
| `run_detection.py` | Loads the queries, runs them against PostgreSQL, and checks recall/precision against the seeded ground truth |
| `setup_database.sh` | One-shot script to create the database, tables, and load the generated CSVs |

## Result

Running detection against the seeded cluster: **100% recall, 90% precision** in the top-20 ranked accounts. Every seeded structuring/mule account was correctly surfaced and ranked at the very top, all correctly identified in the lower-risk category, the segment structurally most exposed to this kind of exploitation since it receives lighter monitoring by design.

## How to run it

Requires PostgreSQL and Python 3.

```bash
pip install -r requirements.txt
python generate_data.py          # generates accounts.csv, transactions.csv, seeded_cluster_ground_truth.csv
./setup_database.sh              # creates the database and loads the CSVs
python run_detection.py          # runs detection, prints the ranked results and recall/precision
```

## A note on the synthetic data

The risk-category distribution and its correlation with KYC completeness are illustrative modelling choices, not sourced from any public statistic. They exist to test the detection logic, not to model real demographics. The regulatory thresholds and categories themselves are real (GwG Annexes 1 and 2, BaFin interpretation guidance).

## Related project

This pipeline's risk-segmentation logic also extends an existing project: [ab-test-checkout-fraud](https://github.com/basseat/ab-test-checkout-fraud), which tests whether a checkout-fraud A/B test's effect differs significantly by risk category.
