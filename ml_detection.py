"""
ML-based structuring/mule detection, compared honestly against the
rules-based SQL pipeline (100% recall, 90% precision in the top-20).

Two approaches, on purpose:
1. Supervised (Random Forest, class-weighted): can a model learn the
   pattern from labels? Evaluated with PR-AUC and stratified k-fold,
   given the severe class imbalance (18 positives / 5000 accounts).
2. Unsupervised (Isolation Forest): can anomaly detection flag the
   same accounts WITHOUT ever seeing labels? This is the more
   realistic real-world scenario for a novel typology nobody has
   confirmed yet.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, classification_report
)

df = pd.read_csv("ml_features.csv")

feature_cols = [
    "max_7d_inflow_count", "avg_7d_inflow_amount", "total_inflows",
    "total_inflow_amount", "avg_hours_to_outflow", "fast_pairs",
    "counterparty_link_count", "device_link_count", "is_lower_risk",
    "kyc_documentation_completeness",
]
X = df[feature_cols].fillna(0)
y = df["label"]

print("=" * 70)
print(f"Dataset: {len(df)} accounts, {y.sum()} positive ({y.mean():.2%})")
print("Severe class imbalance, worth naming explicitly rather than hiding it.")
print("=" * 70)

# ------------------------------------------------------------------
# 1. SUPERVISED: Random Forest, class-weighted, stratified 5-fold CV
# ------------------------------------------------------------------
print("\n--- Supervised: Random Forest (class_weight='balanced') ---")
clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# out-of-fold predicted probabilities, so every account is scored by a
# model that never saw it during training, this is the honest way to
# evaluate given how few positives exist
oof_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]

pr_auc = average_precision_score(y, oof_proba)
print(f"Out-of-fold PR-AUC: {pr_auc:.3f}  (baseline for this imbalance ratio: {y.mean():.3f})")

# top-20 by predicted probability, same comparison basis as the SQL pipeline
df["rf_proba"] = oof_proba
top20_rf = df.nlargest(20, "rf_proba")
recall_rf = top20_rf["label"].sum() / y.sum()
precision_rf = top20_rf["label"].sum() / 20
print(f"Top-20 by RF probability: recall={recall_rf:.0%}, precision={precision_rf:.0%}")

# ------------------------------------------------------------------
# 2. UNSUPERVISED: Isolation Forest, never sees labels
# ------------------------------------------------------------------
print("\n--- Unsupervised: Isolation Forest (no labels used) ---")
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
iso.fit(X)
# lower score = more anomalous
anomaly_score = -iso.score_samples(X)
df["iso_score"] = anomaly_score

top20_iso = df.nlargest(20, "iso_score")
recall_iso = top20_iso["label"].sum() / y.sum()
precision_iso = top20_iso["label"].sum() / 20
print(f"Top-20 by anomaly score: recall={recall_iso:.0%}, precision={precision_iso:.0%}")

# ------------------------------------------------------------------
# 3. HONEST COMPARISON vs the rules-based SQL pipeline
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("COMPARISON: rules-based SQL vs supervised RF vs unsupervised Isolation Forest")
print("=" * 70)
print(f"{'Approach':<35}{'Recall (top-20)':<20}{'Precision (top-20)'}")
print(f"{'Rules-based SQL (existing)':<35}{'100%':<20}{'90%'}")
print(f"{'Supervised Random Forest':<35}{recall_rf:<20.0%}{precision_rf:.0%}")
print(f"{'Unsupervised Isolation Forest':<35}{recall_iso:<20.0%}{precision_iso:.0%}")

df.to_csv("ml_results.csv", index=False)
