import os
import pandas as pd
import matplotlib.pyplot as plt

# Visualize features importances based on gain performances for HLD models

# ─── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH   = './trained_models_lightgbm/features/hld/top_features_by_descriptor.csv'
OUTPUT_DIR = './trained_models_lightgbm/features/hld/visualizations_top10'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1) Read the top‐10 CSV into a “long” DataFrame ─────────────────────────────
df = pd.read_csv(CSV_PATH)

records = []
for _, row in df.iterrows():
    desc = row['descriptor']
    for i in range(1, 11):
        feat = row[f'feat_{i}']
        val  = row[f'value_{i}']
        records.append({
            'feature':    feat,
            'descriptor': desc,
            'importance': float(val)
        })

long_df = pd.DataFrame(records)

# ─── 2) Pivot to a “wide” matrix and fill missing with zero ────────────────────
wide_df = long_df.pivot_table(
    index='feature',
    columns='descriptor',
    values='importance',
    fill_value=0
)

# ─── 3) Heatmap of Top‐10 Importances (zeros included) ────────────────────────
plt.figure(figsize=(24, max(4, 0.3 * len(wide_df))))
plt.imshow(wide_df.values, aspect='auto', cmap='viridis', interpolation='nearest')
plt.xticks(range(len(wide_df.columns)), wide_df.columns, rotation=60, ha='right')
plt.yticks(range(len(wide_df.index)), wide_df.index, fontsize=8)
plt.colorbar(label='Gain Importance (from top 10)')
plt.title('Top-10 Gain Importances Across Descriptors')
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, 'heatmap_top10_zeros.png')
plt.savefig(heatmap_path)
plt.show()
plt.close()
print(f"Saved heatmap → {heatmap_path}")

# ─── 4) Feature-Frequency Bar Chart ─────────────────────────────────────────────
# Count how many descriptors each feature appears in (i.e. importance > 0)
frequency = (wide_df > 0).sum(axis=1).sort_values(ascending=False)

plt.figure(figsize=(12, max(4, 0.2 * len(frequency))))
frequency.plot(kind='bar')
plt.ylabel('Number of descriptors with nonzero importance')
plt.title('Feature Frequency in Top-10 Lists')
plt.tight_layout()

barchart_path = os.path.join(OUTPUT_DIR, 'feature_frequency_top10_zero.png')
plt.savefig(barchart_path)
plt.show()
plt.close()
print(f"Saved feature-frequency bar chart → {barchart_path}")
