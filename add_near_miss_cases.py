"""
Stress-test extension: adds near-miss (hard) cases and jurisdiction risk,
so rules-based, supervised, and unsupervised approaches can actually be
compared meaningfully, rather than all trivially finding an obvious pattern.

Near-miss cases: structuring-like behaviour that's deliberately harder to
catch, wider amount variance, slower pass-through, weaker device overlap,
some legitimate-looking transactions mixed in. These simulate a more
sophisticated ring, or early-stage, partially-obfuscated activity.

Jurisdiction risk: counterparty_country mapped to a simple high/low risk
classification, echoing the geospatial/ML address risk classification
line from the real job description of this kind of role.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(99)

accounts = pd.read_csv("accounts.csv")
transactions = pd.read_csv("transactions.csv")
truth = pd.read_csv("seeded_cluster_ground_truth.csv")

START = datetime(2026, 1, 1)
N_DAYS = 90

# --- Jurisdiction risk mapping (illustrative, echoes FATF-style categorisation) ---
# NOT sourced from the live FATF list, a simplified illustrative mapping for
# feature engineering purposes only, would need to be kept current in production.
HIGH_RISK_COUNTRIES = {"KW", "PG", "IR", "KP", "MM"}  # FATF grey/black-list style examples
LOW_RISK_COUNTRIES = {"DE", "AT", "FR", "NL"}

def jurisdiction_risk(country):
    if country in HIGH_RISK_COUNTRIES:
        return "high"
    elif country in LOW_RISK_COUNTRIES:
        return "low"
    return "medium"

transactions["jurisdiction_risk"] = transactions["counterparty_country"].apply(jurisdiction_risk)

# --- Seed a harder, near-miss cluster: 15 accounts, deliberately noisier ---
lower_risk_ids = accounts.loc[accounts.initial_risk_category == "lower_risk_sdd", "account_id"].tolist()
existing_cluster = set(truth.account_id)
available_ids = [a for a in lower_risk_ids if a not in existing_cluster]
near_miss_ids = rng.choice(available_ids, size=15, replace=False)

near_miss_device_pool = [f"NEARMISS_DEV_{i}" for i in range(4)]  # weaker device overlap than the obvious cluster
near_miss_beneficiary_pool = [f"NEARMISS_BEN_{i}" for i in range(6)]  # more spread out

accounts.loc[accounts.account_id.isin(near_miss_ids), "onboarding_device_id"] = \
    rng.choice(near_miss_device_pool, size=15)

records = []
tx_id_start = 900000
tx_id = tx_id_start
for acc in near_miss_ids:
    n_days_active = rng.integers(15, N_DAYS)
    active_days = rng.choice(range(N_DAYS), size=n_days_active, replace=False)
    for d in active_days:
        date = START + timedelta(days=int(d))
        # noisier structuring: wider amount range, sometimes above the clean threshold,
        # fewer transactions per day, slower and less consistent pass-through
        n_tx_today = rng.integers(1, 4)
        for _ in range(n_tx_today):
            tx_id += 1
            amount = rng.uniform(80, 1400)  # sometimes crosses the €1,000 threshold, unlike the obvious cluster
            counterparty = rng.choice(near_miss_beneficiary_pool)
            country = rng.choice(["DE", "PG", "AT"], p=[0.5, 0.2, 0.3])
            records.append([f"T{tx_id:08d}", acc, "inflow", round(amount, 2),
                             counterparty, country, "transfer", date])
            if rng.random() < 0.6:  # only sometimes shows fast pass-through, unlike the obvious cluster
                tx_id += 1
                delay_hours = rng.integers(1, 30)  # slower and more variable than the obvious cluster
                records.append([f"T{tx_id:08d}", acc, "outflow", round(amount * rng.uniform(0.85, 0.98), 2),
                                 counterparty, country, "transfer", date + timedelta(hours=int(delay_hours))])
        # occasionally mix in a legitimate-looking transaction to add noise
        if rng.random() < 0.3:
            tx_id += 1
            records.append([f"T{tx_id:08d}", acc, "outflow", round(rng.lognormal(6.0, 0.8), 2),
                             f"CP{rng.integers(0, 20000):06d}", "DE", "card", date])

near_miss_tx = pd.DataFrame(records, columns=[
    "transaction_id", "account_id", "direction", "amount",
    "counterparty_id", "counterparty_country", "channel", "timestamp"
])
near_miss_tx["jurisdiction_risk"] = near_miss_tx["counterparty_country"].apply(jurisdiction_risk)

transactions = pd.concat([transactions, near_miss_tx], ignore_index=True)

# extended ground truth: obvious cluster + near-miss cluster, kept separate for evaluation
near_miss_truth = pd.DataFrame({"account_id": near_miss_ids})
near_miss_truth.to_csv("near_miss_ground_truth.csv", index=False)

accounts.to_csv("accounts.csv", index=False)
transactions.to_csv("transactions.csv", index=False)

print(f"Added {len(near_miss_ids)} near-miss accounts, {len(near_miss_tx)} new transactions")
print(f"Total transactions now: {len(transactions)}")
print(f"Jurisdiction risk distribution:\n{transactions.jurisdiction_risk.value_counts()}")
