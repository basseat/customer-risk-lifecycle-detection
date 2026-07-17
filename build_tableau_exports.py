"""
Tableau-ready exports. Each CSV is in tidy/long format, one row per
data point, ready to drag straight into Tableau without further
reshaping.
"""
import pandas as pd

df = pd.read_csv("ml_results.csv")
truth = pd.read_csv("seeded_cluster_ground_truth.csv")
near_miss = pd.read_csv("near_miss_ground_truth.csv")
obvious_ids = set(truth.account_id)
nearmiss_ids = set(near_miss.account_id)

# ------------------------------------------------------------------
# 1. Method comparison: recall by method and cluster type (tidy format)
#    -> Tableau: grouped bar chart, Method on columns, Cluster on color
# ------------------------------------------------------------------
K = len(obvious_ids) + len(nearmiss_ids)
rows = []
for col, method in [("rf_proba", "Supervised Random Forest"), ("iso_score", "Unsupervised Isolation Forest")]:
    topk = df.nlargest(K, col)
    obv_hit = len(set(topk.account_id) & obvious_ids)
    nm_hit = len(set(topk.account_id) & nearmiss_ids)
    rows.append({"method": method, "cluster_type": "Obvious cluster", "recall": obv_hit / len(obvious_ids)})
    rows.append({"method": method, "cluster_type": "Near-miss cluster", "recall": nm_hit / len(nearmiss_ids)})
    rows.append({"method": method, "cluster_type": "Overall", "recall": (obv_hit + nm_hit) / K})
# rules-based, from the known SQL result
rows.append({"method": "Rules-based SQL", "cluster_type": "Obvious cluster", "recall": 1.0})
rows.append({"method": "Rules-based SQL", "cluster_type": "Near-miss cluster", "recall": 0.0})
rows.append({"method": "Rules-based SQL", "cluster_type": "Overall", "recall": 18/33})

method_comparison = pd.DataFrame(rows)
method_comparison.to_csv("tableau_method_comparison.csv", index=False)

# ------------------------------------------------------------------
# 2. Feature importance (tidy format)
#    -> Tableau: horizontal bar chart, sorted descending
# ------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
feature_cols = [
    "max_7d_inflow_count", "avg_7d_inflow_amount", "total_inflows",
    "total_inflow_amount", "avg_hours_to_outflow", "fast_pairs",
    "counterparty_link_count", "device_link_count", "is_lower_risk",
    "kyc_documentation_completeness", "high_risk_country_share",
]
X = df[feature_cols].fillna(0)
y = df["label"]
clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(X, y)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": clf.feature_importances_,
}).sort_values("importance", ascending=False)
importance_df.to_csv("tableau_feature_importance.csv", index=False)

# ------------------------------------------------------------------
# 3. Score distribution: every account's RF score, labelled by group
#    -> Tableau: histogram or box plot, coloured by group
# ------------------------------------------------------------------
def label_group(row):
    if row["account_id"] in obvious_ids:
        return "True positive (obvious)"
    elif row["account_id"] in nearmiss_ids:
        return "True positive (near-miss)"
    return "Normal account"

score_dist = df[["account_id", "rf_proba", "iso_score"]].copy()
score_dist["group"] = df.apply(label_group, axis=1)
score_dist.to_csv("tableau_score_distribution.csv", index=False)

# ------------------------------------------------------------------
# 4. Risk category breakdown of flagged vs. normal accounts (tidy format)
#    -> Tableau: stacked bar or 100% stacked bar
# ------------------------------------------------------------------
top33 = df.nlargest(K, "rf_proba")
flagged_by_category = top33.initial_risk_category.value_counts().rename_axis("risk_category").reset_index(name="count")
flagged_by_category["group"] = "Flagged (top 33 by RF score)"

all_by_category = df.initial_risk_category.value_counts().rename_axis("risk_category").reset_index(name="count")
all_by_category["group"] = "All accounts"

risk_breakdown = pd.concat([flagged_by_category, all_by_category], ignore_index=True)
risk_breakdown.to_csv("tableau_risk_category_breakdown.csv", index=False)

print("Tableau exports written:")
for f in ["tableau_method_comparison.csv", "tableau_feature_importance.csv",
          "tableau_score_distribution.csv", "tableau_risk_category_breakdown.csv"]:
    d = pd.read_csv(f)
    print(f"  {f}: {d.shape[0]} rows")
