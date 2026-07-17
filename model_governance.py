"""
Auditability layer: can this pipeline's results be trusted, checked, and
explained after the fact, not just today. Three practical questions this
covers: can someone reproduce the exact same numbers later, is
performance tracked over time rather than only known at the moment of
one run, and does the review cadence match the actual regulatory
obligations the underlying risk categories carry.
"""
import json
import hashlib
from datetime import datetime, timezone
import pandas as pd

# ------------------------------------------------------------------
# 1. CAN THIS BE RE-RUN AND VERIFIED LATER
# ------------------------------------------------------------------
# Every stochastic step in this project (data generation, near-miss
# seeding, model training, cross-validation splits) uses a fixed random
# seed. That means re-running generate_data.py -> add_near_miss_cases.py
# -> build_features.py -> ml_detection.py end to end produces bit-for-bit
# identical results, not just "similar" ones. That's the actual test of
# reproducibility: can someone else, or you in six months, get the exact
# same numbers back.
SEEDS_USED = {
    "generate_data.py": 42,
    "add_near_miss_cases.py": 99,
    "ml_detection.py (RandomForestClassifier, StratifiedKFold)": 42,
    "ml_detection.py (IsolationForest)": 42,
}

# ------------------------------------------------------------------
# 2. TRACKING RESULTS OVER TIME, NOT JUST ONE SNAPSHOT
# ------------------------------------------------------------------
# A minimal model registry: every time the model is evaluated, log the
# metrics with a timestamp and a hash of the feature set used, so
# performance can be tracked over time and any drift or regression is
# visible, rather than only ever knowing the latest number.
def hash_feature_list(feature_cols):
    return hashlib.sha256(",".join(sorted(feature_cols)).encode()).hexdigest()[:12]

feature_cols = [
    "max_7d_inflow_count", "avg_7d_inflow_amount", "total_inflows",
    "total_inflow_amount", "avg_hours_to_outflow", "fast_pairs",
    "counterparty_link_count", "device_link_count", "is_lower_risk",
    "kyc_documentation_completeness", "high_risk_country_share",
]

df = pd.read_csv("ml_results.csv")
truth = pd.read_csv("seeded_cluster_ground_truth.csv")
near_miss = pd.read_csv("near_miss_ground_truth.csv")
positive_ids = set(truth.account_id) | set(near_miss.account_id)
K = len(positive_ids)

top_k_rf = df.nlargest(K, "rf_proba")
top_k_iso = df.nlargest(K, "iso_score")

registry_entry = {
    "run_timestamp": datetime.now(timezone.utc).isoformat(),
    "feature_set_hash": hash_feature_list(feature_cols),
    "feature_count": len(feature_cols),
    "dataset_size": len(df),
    "positive_count": len(positive_ids),
    "seeds": SEEDS_USED,
    "models": {
        "random_forest": {
            "recall_at_k": round(len(set(top_k_rf.account_id) & positive_ids) / K, 3),
            "precision_at_k": round(len(set(top_k_rf.account_id) & positive_ids) / K, 3),
        },
        "isolation_forest": {
            "recall_at_k": round(len(set(top_k_iso.account_id) & positive_ids) / K, 3),
            "precision_at_k": round(len(set(top_k_iso.account_id) & positive_ids) / K, 3),
        },
    },
}

# append to a running log rather than overwrite, this is what makes
# performance trackable over time rather than a single snapshot
try:
    with open("model_registry.json") as f:
        registry = json.load(f)
except FileNotFoundError:
    registry = []
registry.append(registry_entry)
with open("model_registry.json", "w") as f:
    json.dump(registry, f, indent=2)

print("Model registry entry logged:")
print(json.dumps(registry_entry, indent=2))

# ------------------------------------------------------------------
# 3. MATCHING REVIEW CADENCE TO ACTUAL REGULATORY OBLIGATIONS
# ------------------------------------------------------------------
print("""
Review cadence notes (tied to the actual GwG obligations, not a generic best-practice checklist):

- Model retraining/review cadence should align with the GwG-mandated
  risk-category review cycles this project's schema is built on:
  higher-risk accounts reviewed at least annually, normal-risk at least
  every 5 years. A detection model influencing risk classification
  should be reviewed at least as often as the riskiest category it
  affects, i.e. at minimum annually.
- A SAR filing automatically triggers a 21-day (ML) or 6-month (TF)
  heightened-risk window under current BaFin guidance. Any model whose
  output could contribute to a SAR decision needs its scoring logic
  and feature set to be documented and reproducible for that period,
  in case of regulatory review, which is exactly what the seed logging
  and feature-hash logging above are for.
- Feature importance transparency (see false_positive_analysis.py)
  matters beyond model quality, it's what lets an analyst explain to
  an investigator, or eventually a regulator, WHY an account was
  flagged, not just that it was. A model that can't explain its own
  top features is a harder model to defend under audit.
""")
