"""
A/B Test: Checkout Fraud, extended with risk-category segmentation.
Tests whether the fraud increase from the simplified checkout is
disproportionately concentrated in lower-risk (SDD) accounts,
mirroring the real 9PSB Tier 1 finding.
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

rng = np.random.default_rng(7)

N = 24000  # matches the original project's scale (24k users, 4 markets)

# risk category distribution, same shape as the CRL dataset
risk_category = rng.choice(["lower_risk_sdd", "normal_risk_cdd", "higher_risk_edd"],
                            size=N, p=[0.15, 0.75, 0.10])
treatment = rng.choice([0, 1], size=N)  # 0 = control, 1 = treatment (simplified checkout)

# Baseline fraud probability by risk category (lower-risk = lighter monitoring = higher baseline vulnerability)
base_fraud_p = np.select(
    [risk_category == "lower_risk_sdd", risk_category == "normal_risk_cdd", risk_category == "higher_risk_edd"],
    [0.025, 0.018, 0.012]
)

# Treatment effect: the simplified checkout disproportionately raises fraud
# risk in lower-risk accounts (the hypothesis), moderate lift elsewhere.
treatment_lift = np.select(
    [risk_category == "lower_risk_sdd", risk_category == "normal_risk_cdd", risk_category == "higher_risk_edd"],
    [0.032, 0.014, 0.006]
)

fraud_p = base_fraud_p + treatment * treatment_lift
fraud = rng.binomial(1, fraud_p)

df = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(N)],
    "risk_category": risk_category,
    "treatment": treatment,
    "fraud": fraud
})

print("=" * 70)
print("1. OVERALL EFFECT (matches original project's headline finding)")
print("=" * 70)
ct = pd.crosstab(df.treatment, df.fraud)
control_rate = df[df.treatment == 0].fraud.mean()
treat_rate = df[df.treatment == 1].fraud.mean()
n_c, n_t = (df.treatment == 0).sum(), (df.treatment == 1).sum()
count = np.array([df[df.treatment == 1].fraud.sum(), df[df.treatment == 0].fraud.sum()])
nobs = np.array([n_t, n_c])
from statsmodels.stats.proportion import proportions_ztest
z, p = proportions_ztest(count, nobs)
print(f"Control fraud rate:   {control_rate:.4f}")
print(f"Treatment fraud rate: {treat_rate:.4f}")
print(f"z = {z:.3f}, p = {p:.5f}")

print()
print("=" * 70)
print("2. SEGMENTED BY RISK CATEGORY (the new analysis)")
print("=" * 70)
for cat in ["lower_risk_sdd", "normal_risk_cdd", "higher_risk_edd"]:
    sub = df[df.risk_category == cat]
    c_rate = sub[sub.treatment == 0].fraud.mean()
    t_rate = sub[sub.treatment == 1].fraud.mean()
    count = np.array([sub[sub.treatment == 1].fraud.sum(), sub[sub.treatment == 0].fraud.sum()])
    nobs = np.array([(sub.treatment == 1).sum(), (sub.treatment == 0).sum()])
    z, p = proportions_ztest(count, nobs)
    lift_pp = (t_rate - c_rate) * 100
    print(f"{cat:18s}  control={c_rate:.4f}  treatment={t_rate:.4f}  "
          f"lift={lift_pp:+.2f}pp  z={z:.3f}  p={p:.5f}")

print()
print("=" * 70)
print("3. LOGISTIC REGRESSION WITH TREATMENT x RISK_CATEGORY INTERACTION")
print("   Tests whether the treatment effect genuinely DIFFERS by segment")
print("=" * 70)
df["risk_category"] = pd.Categorical(df.risk_category,
                                      categories=["normal_risk_cdd", "lower_risk_sdd", "higher_risk_edd"])
model = smf.logit("fraud ~ treatment * C(risk_category)", data=df).fit(disp=0)
print(model.summary())

print()
print("=" * 70)
print("4. INTERPRETATION")
print("=" * 70)
interaction_term = "treatment:C(risk_category)[T.lower_risk_sdd]"
if interaction_term in model.pvalues.index:
    p_int = model.pvalues[interaction_term]
    coef = model.params[interaction_term]
    print(f"Interaction coefficient (treatment x lower_risk_sdd): {coef:.4f}, p = {p_int:.5f}")
    if p_int < 0.05 and coef > 0:
        print(">> SUPPORTED: the treatment effect on fraud is significantly LARGER")
        print("   for lower-risk (SDD) accounts than for the normal-risk baseline.")
        print("   The fraud increase from the simplified checkout is disproportionately")
        print("   concentrated in the segment that receives lighter monitoring by design.")
    else:
        print(">> NOT SUPPORTED at p<0.05: no significant interaction found in this run.")

df.to_csv("/home/claude/crl_project/ab_test_risk_segmented.csv", index=False)
