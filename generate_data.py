"""
Customer Risk Lifecycle: synthetic data generator
Schema follows the GwG-based event taxonomy (lower/normal/higher risk categories).
Seeds a deliberate structuring / mule-network cluster for detection to find.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

N_ACCOUNTS = 5000
N_DAYS = 90
START = datetime(2026, 1, 1)

# ---- 1. accounts ----
risk_probs = [0.15, 0.75, 0.10]  # lower, normal, higher
risk_categories = rng.choice(["lower_risk_sdd", "normal_risk_cdd", "higher_risk_edd"],
                              size=N_ACCOUNTS, p=risk_probs)

accounts = pd.DataFrame({
    "account_id": [f"A{i:05d}" for i in range(N_ACCOUNTS)],
    "initial_risk_category": risk_categories,
    "kyc_documentation_completeness": np.clip(
        rng.normal(loc=np.where(risk_categories == "lower_risk_sdd", 0.65,
                    np.where(risk_categories == "higher_risk_edd", 0.95, 0.85)),
                    scale=0.08), 0.3, 1.0),
    "onboarding_channel": rng.choice(["app", "branch", "partner_institution"], size=N_ACCOUNTS, p=[0.8, 0.15, 0.05]),
    "onboarding_device_id": [f"DEV{i:06d}" for i in range(N_ACCOUNTS)],  # unique per account by default
    "opened_at": [START - timedelta(days=int(d)) for d in rng.integers(30, 1500, size=N_ACCOUNTS)],
})

# ---- 2. seed a structuring / mule cluster among lower-risk accounts ----
lower_risk_ids = accounts.loc[accounts.initial_risk_category == "lower_risk_sdd", "account_id"].tolist()
CLUSTER_SIZE = 18
cluster_ids = rng.choice(lower_risk_ids, size=CLUSTER_SIZE, replace=False)
# realistic German/EU network signal: multiple "unrelated" identities completing
# remote video-ident onboarding (IDnow/WebID-style) from the same small pool of
# devices, a well-documented mule-recruitment indicator, e.g. a recruiter having
# several recruits complete onboarding on the same phone or laptop.
shared_device_pool = [f"MULE_DEV_{i}" for i in range(2)]
accounts.loc[accounts.account_id.isin(cluster_ids), "onboarding_device_id"] = \
    rng.choice(shared_device_pool, size=CLUSTER_SIZE)
shared_beneficiary_pool = [f"MULE_BEN_{i}" for i in range(3)]  # a small set of shared payout accounts

# ---- 3. transactions ----
records = []
tx_id = 0
for _, row in accounts.iterrows():
    acc = row.account_id
    is_cluster = acc in cluster_ids
    n_days_active = rng.integers(20, N_DAYS)
    active_days = rng.choice(range(N_DAYS), size=n_days_active, replace=False)

    for d in active_days:
        date = START + timedelta(days=int(d))
        if is_cluster:
            # structuring pattern: many small inflows, high frequency
            n_tx_today = rng.integers(3, 8)
            for _ in range(n_tx_today):
                tx_id += 1
                amount = rng.uniform(80, 950)  # small EUR amounts, deliberately kept under the €1,000 GwG transfer-information threshold
                counterparty = rng.choice(shared_beneficiary_pool)
                records.append([f"T{tx_id:07d}", acc, "inflow", round(amount, 2),
                                 counterparty, "DE", "transfer", date])
                # rapid pass-through: outflow within hours, same day
                tx_id += 1
                records.append([f"T{tx_id:07d}", acc, "outflow", round(amount * rng.uniform(0.92, 0.99), 2),
                                 counterparty, "DE", "transfer", date + timedelta(hours=int(rng.integers(1, 6)))])
        else:
            # normal behaviour: occasional, varied-size transactions
            n_tx_today = rng.integers(1, 3)
            for _ in range(n_tx_today):
                tx_id += 1
                direction = rng.choice(["inflow", "outflow"])
                amount = rng.lognormal(mean=6.5, sigma=1.0)
                counterparty = f"CP{rng.integers(0, 20000):06d}"
                country = rng.choice(["DE", "AT", "FR", "NL", "ES"], p=[0.7, 0.1, 0.1, 0.05, 0.05])
                records.append([f"T{tx_id:07d}", acc, direction, round(amount, 2),
                                 counterparty, country, rng.choice(["transfer", "card", "atm"]), date])

transactions = pd.DataFrame(records, columns=[
    "transaction_id", "account_id", "direction", "amount",
    "counterparty_id", "counterparty_country", "channel", "timestamp"
])

# ---- 4. save ----
accounts.to_csv("accounts.csv", index=False)
transactions.to_csv("transactions.csv", index=False)
# ground-truth list for validating detection, kept separate so detection queries
# never see it directly
pd.DataFrame({"account_id": cluster_ids}).to_csv("seeded_cluster_ground_truth.csv", index=False)

print(f"Accounts: {len(accounts)}")
print(f"Transactions: {len(transactions)}")
print(f"Seeded cluster accounts: {len(cluster_ids)}")
print(accounts.initial_risk_category.value_counts())
