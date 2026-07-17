"""
Feature engineering for ML-based structuring/mule detection.
Builds a per-account feature table from the same signals the rules-based
SQL pipeline uses (velocity, pass-through, network links), so the ML
approach and the rules-based approach are compared on equal footing.
"""
import pandas as pd
import numpy as np

accounts = pd.read_csv("accounts.csv")
accounts["opened_at"] = pd.to_datetime(accounts["opened_at"])
transactions = pd.read_csv("transactions.csv")
transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], format="mixed")
truth = pd.read_csv("seeded_cluster_ground_truth.csv")
near_miss_truth = pd.read_csv("near_miss_ground_truth.csv")
cluster_ids = set(truth.account_id) | set(near_miss_truth.account_id)
near_miss_ids = set(near_miss_truth.account_id)

inflows = transactions[transactions.direction == "inflow"].copy()
outflows = transactions[transactions.direction == "outflow"].copy()

# --- Velocity features: rolling 7-day inflow stats, summarised per account ---
inflows = inflows.sort_values(["account_id", "timestamp"])
velocity_rows = []
for acc_id, grp in inflows.groupby("account_id"):
    grp = grp.set_index("timestamp")
    counts = grp["amount"].rolling("7D").count()
    means = grp["amount"].rolling("7D").mean()
    velocity_rows.append({
        "account_id": acc_id,
        "max_7d_inflow_count": counts.max(),
        "avg_7d_inflow_amount": means.mean(),
        "total_inflows": len(grp),
        "total_inflow_amount": grp["amount"].sum(),
    })
velocity_df = pd.DataFrame(velocity_rows)

# --- Pass-through features: time from inflow to next outflow, same account ---
pass_through_rows = []
for acc_id, in_grp in inflows.groupby("account_id"):
    out_grp = outflows[outflows.account_id == acc_id]
    if out_grp.empty:
        pass_through_rows.append({"account_id": acc_id, "avg_hours_to_outflow": np.nan, "fast_pairs": 0})
        continue
    gaps = []
    for _, row in in_grp.iterrows():
        candidates = out_grp[(out_grp.timestamp > row.timestamp) &
                              (out_grp.timestamp <= row.timestamp + pd.Timedelta(hours=24))]
        if not candidates.empty:
            gap_hours = (candidates.timestamp.min() - row.timestamp).total_seconds() / 3600
            gaps.append(gap_hours)
    pass_through_rows.append({
        "account_id": acc_id,
        "avg_hours_to_outflow": np.mean(gaps) if gaps else np.nan,
        "fast_pairs": sum(1 for g in gaps if g < 12),
    })
pass_through_df = pd.DataFrame(pass_through_rows)

# --- Network features: counterparty overlap and device overlap counts ---
cp_counts = transactions.groupby("counterparty_id")["account_id"].nunique()
shared_cps = cp_counts[cp_counts > 1].index
tx_shared = transactions[transactions.counterparty_id.isin(shared_cps)]
counterparty_link_count = tx_shared.groupby("account_id")["counterparty_id"].nunique().rename("counterparty_link_count")

device_counts = accounts.groupby("onboarding_device_id")["account_id"].transform("count")
device_link_df = accounts[["account_id"]].copy()
device_link_df["device_link_count"] = np.where(device_counts > 1, device_counts, 0)

# --- Jurisdiction risk feature: proportion of high-risk-country counterparties ---
HIGH_RISK_COUNTRIES = {"KW", "PG", "IR", "KP", "MM"}
jurisdiction_stats = transactions.groupby("account_id")["counterparty_country"].agg(
    high_risk_country_share=lambda x: x.isin(HIGH_RISK_COUNTRIES).mean()
).reset_index()

# --- Assemble feature table ---
features = accounts[["account_id", "initial_risk_category", "kyc_documentation_completeness"]].copy()
features = features.merge(velocity_df, on="account_id", how="left")
features = features.merge(pass_through_df, on="account_id", how="left")
features = features.merge(counterparty_link_count, on="account_id", how="left")
features = features.merge(device_link_df, on="account_id", how="left")
features = features.merge(jurisdiction_stats, on="account_id", how="left")

features["max_7d_inflow_count"] = features["max_7d_inflow_count"].fillna(0)
features["avg_7d_inflow_amount"] = features["avg_7d_inflow_amount"].fillna(0)
features["total_inflows"] = features["total_inflows"].fillna(0)
features["total_inflow_amount"] = features["total_inflow_amount"].fillna(0)
features["avg_hours_to_outflow"] = features["avg_hours_to_outflow"].fillna(999)  # no pass-through observed
features["fast_pairs"] = features["fast_pairs"].fillna(0)
features["counterparty_link_count"] = features["counterparty_link_count"].fillna(0)
features["device_link_count"] = features["device_link_count"].fillna(0)
features["high_risk_country_share"] = features["high_risk_country_share"].fillna(0)
features["is_lower_risk"] = (features["initial_risk_category"] == "lower_risk_sdd").astype(int)

features["label"] = features["account_id"].isin(cluster_ids).astype(int)
features["is_near_miss"] = features["account_id"].isin(near_miss_ids).astype(int)

features.to_csv("ml_features.csv", index=False)
print(f"Feature table: {len(features)} accounts, {features['label'].sum()} positive (structuring/mule)")
print(features.groupby("label")[["max_7d_inflow_count", "fast_pairs", "counterparty_link_count", "device_link_count"]].mean())
