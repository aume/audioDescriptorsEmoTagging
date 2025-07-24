import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# The goal is to confirm that 600_Sounds and 613_MixedSounds really occupy different regions
# of feature space so we can safely pool them as extra data points without leaking the same
# patterns into both train and test.

# ─── 1. LOAD & LABEL ────────────────────────────────────────────────────────
df600 = pd.read_csv("./600_Sounds_features.csv")
df613 = pd.read_csv("./613_MixedSounds_features.csv")
df600["group"] = 0    # 600_Sounds
df613["group"] = 1    # 613_MixedSounds
df = pd.concat([df600, df613], ignore_index=True)

# ─── 2. DEFINE FEATURE COLUMNS ─────────────────────────────────────────────
meta_cols = {"Filename", "RelPath", "group"}
features = [c for c in df.columns if c not in meta_cols]
print(f"Total files: {len(df)}, Features: {len(features)}")

# ─── 3. UNIVARIATE SIGNIFICANCE TESTS ──────────────────────────────────────
pvals = []
for f in features:
    a = df.loc[df.group==0, f].dropna()
    b = df.loc[df.group==1, f].dropna()
    p = mannwhitneyu(a, b, alternative="two-sided").pvalue
    pvals.append(p)

pvals = np.array(pvals)
reject, qvals, _, _ = multipletests(pvals, method="fdr_bh")
sig_features = np.array(features)[reject]
print(f"{len(sig_features)} features survive FDR<0.05:", sig_features)

# ─── 3b. VISUALIZE BH CORRECTION ────────────────────────────────────────────
alpha = 0.05
m = len(pvals)
# sort raw p-values
sorted_p = np.sort(pvals)
# BH threshold line: (i/m)*alpha
bh_threshold = (np.arange(1, m+1) / m) * alpha

plt.figure(figsize=(8,6))
plt.plot(np.arange(1, m+1), sorted_p, marker='o', linestyle='', label='Sorted p-values')
plt.plot(np.arange(1, m+1), bh_threshold, color='red', label=f'BH line (α={alpha})')
plt.xlabel("Index (sorted tests)")
plt.ylabel("p-value")
plt.title("Benjamini–Hochberg FDR Correction")
plt.legend()
plt.tight_layout()
plt.show()

# ─── 4. CLASSIFICATION CHECK (“600 vs 613”) ────────────────────────────────
X = df[features].fillna(0)
y = df["group"]
clf = RandomForestClassifier(n_estimators=200, random_state=0)
scores = cross_val_score(clf, X, y,
                         cv=StratifiedKFold(5),
                         scoring="accuracy")
print("Group separability accuracy (RF):", scores.mean())