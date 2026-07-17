"""
Alert quality review: which normal accounts does the model rank
suspiciously high, and why? This matters operationally, every flagged
account costs an investigator's time, so understanding why a model
over-flags is as important as its raw recall and precision.
"""
import pandas as pd
import numpy as np

df = pd.read_csv("ml_results.csv")
truth = pd.read_csv("seeded_cluster_ground_truth.csv")
near_miss = pd.read_csv("near_miss_ground_truth.csv")
positive_ids = set(truth.account_id) | set(near_miss.account_id)

feature_cols = [
    "max_7d_inflow_count", "avg_7d_inflow_amount", "total_inflows",
    "total_inflow_amount", "avg_hours_to_outflow", "fast_pairs",
    "counterparty_link_count", "device_link_count", "is_lower_risk",
    "kyc_documentation_completeness", "high_risk_country_share",
]

# "false positives" here = highest-scored accounts that are NOT actually positive,
# looking past the top-33 (which we already showed is clean) into the next band,
# this is the realistic operational question: what does the model flag that
# an investigator would then clear, and does it cluster around an explainable cause
next_band = df[~df.account_id.isin(positive_ids)].nlargest(30, "rf_proba")

print("=" * 70)
print("ALERT QUALITY REVIEW: top 30 non-positive accounts by RF score")
print("=" * 70)
print(f"Score range in this band: {next_band.rf_proba.min():.3f} to {next_band.rf_proba.max():.3f}")
print(f"True positive score range: {df[df.account_id.isin(positive_ids)].rf_proba.min():.3f} to "
      f"{df[df.account_id.isin(positive_ids)].rf_proba.max():.3f}")

print("\nFeature averages: top non-positive band vs. general population")
comparison = pd.DataFrame({
    "top_band": next_band[feature_cols].mean(),
    "general_population": df[~df.account_id.isin(positive_ids)][feature_cols].mean(),
})
comparison["ratio"] = (comparison["top_band"] / comparison["general_population"].replace(0, np.nan)).round(1)
print(comparison.round(2).to_string())

# feature importance from the trained model, to explain WHY these accounts rank high
from sklearn.ensemble import RandomForestClassifier
X = df[feature_cols].fillna(0)
y = df["account_id"].isin(positive_ids).astype(int)
clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(X, y)
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.round(3).to_string())

# Score-gap check: is this next-highest band a genuinely ambiguous grey zone,
# or just the top of an otherwise uniform low-score group? A gap this large,
# combined with feature ratios near 1.0x, indicates the latter on this
# synthetic dataset. See README for the full reasoning and its limits,
# real ambiguity needs real historical alert data to characterise properly.
score_gap = df[df.account_id.isin(positive_ids)].rf_proba.min() - next_band.rf_proba.max()
print(f"\nScore gap (min true positive - max non-positive): {score_gap:.3f}")

next_band.to_csv("false_positive_band.csv", index=False)
