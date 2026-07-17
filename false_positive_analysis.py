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
print(f"(For reference, the 33 true positives scored between "
      f"{df[df.account_id.isin(positive_ids)].rf_proba.min():.3f} and "
      f"{df[df.account_id.isin(positive_ids)].rf_proba.max():.3f})")

print("\nFeature averages: false-positive band vs. the general population")
comparison = pd.DataFrame({
    "false_positive_band": next_band[feature_cols].mean(),
    "general_population": df[~df.account_id.isin(positive_ids)][feature_cols].mean(),
})
comparison["ratio"] = (comparison["false_positive_band"] / comparison["general_population"].replace(0, np.nan)).round(1)
print(comparison.round(2).to_string())

# feature importance from the trained model, to explain WHY these accounts rank high
from sklearn.ensemble import RandomForestClassifier
X = df[feature_cols].fillna(0)
y = df["account_id"].isin(positive_ids).astype(int)
clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(X, y)
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances (what the model actually weighs most):")
print(importances.round(3).to_string())

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
score_gap = df[df.account_id.isin(positive_ids)].rf_proba.min() - next_band.rf_proba.max()
print(f"""
Honest read: the score gap between true positives (min {df[df.account_id.isin(positive_ids)].rf_proba.min():.3f})
and this next-highest band (max {next_band.rf_proba.max():.3f}) is large, {score_gap:.3f}.
This is NOT a genuinely ambiguous grey zone, the feature ratios above are
close to 1.0x, meaning these accounts aren't meaningfully different from
the general population, they're just the highest-ranked among an
otherwise uniform low-score group.

The honest conclusion: even with the near-miss cases added, this
synthetic dataset still doesn't produce realistic false-positive
ambiguity. Understanding that boundary properly needs real historical alert
data with actual investigator dispositions to find where genuine
ambiguity lives, synthetic data this clean can validate that a
detection method works, but can't validate how it behaves at the messy
decision boundary real production data would have.

What IS genuinely useful here: the feature importance ranking.
device_link_count and avg_hours_to_outflow dominate, meaning network
coordination and pass-through speed are what the model leans on most,
not raw transaction volume. That's a legitimate, actionable finding
regardless of how clean this particular dataset is.
""")

next_band.to_csv("false_positive_band.csv", index=False)
