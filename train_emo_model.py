import os
import glob
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import optuna
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from extractor import Extractor 

# 1. fetch sounds from directory
# fetch rating value from database file
# 2. extract features from sound file
# df = [values,..][[features],..]
# 3. train and evaluate model 

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
n_trials       = 60  # how many Optuna trials to run
build_dataset  = True

# !!! IMPORTANT: Set these paths based on whether you are training Valence or Arousal !!!
# Uncomment one block at a time to train each model:

# For Valence training:
# va_file = './datasets/out_Valence_feature_importance.csv'
# df_ratings_path = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Ratings/Valence.csv"
# model_output_name = "./trained_models_lightgbm/va_model/trained_valence_model_lightgbm.joblib"
# print("--- Configuring for Valence Model Training ---")

# # For Arousal training:
va_file         = './datasets/out_Arousal_feature_importance.csv'
df_ratings_path = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Ratings/Arousal.csv"
model_output_name = "./trained_models_lightgbm/va_model/trained_arousal_model_lightgbm.joblib"
print("--- Configuring for Arousal Model Training ---")

# !!! IMPORTANT: Create a new folder then put 600_Sounds and 613_MixedSounds folders here
audioFolder = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Audio/600Sounds_and_613MixedSounds"

# ─── 0) scan recursively and map basenames to full paths ─────────────────────────
wav_paths = glob.glob(os.path.join(audioFolder, '**', '*.wav'), recursive=True)
file_to_path = { os.path.basename(p): p for p in wav_paths }
print(f"Found {len(wav_paths)} .wav files under {audioFolder}")

# ─── 1) BUILD OR LOAD DATASET ───────────────────────────────────────────────────
if build_dataset:
    print('Building dataset by extracting features...')

    frame_size  = 2048
    hop_size    = 1024
    sample_rate = 32000
    s1 = Extractor(sample_rate, frame_size, hop_size)
    
    all_features_list = []
    df_ratings = pd.read_csv(df_ratings_path, header=None, names=['file','value'])
    
    for _, row in df_ratings.iterrows():
        fname = row['file']
        value = row['value']
        file_path = file_to_path.get(fname)
        
        if file_path:
            print(f"Extracting from: {fname}")
            try:
                features = s1.extract(file_path)
                if features:
                    features['file']  = fname
                    features['value'] = float(value)
                    all_features_list.append(features)
                else:
                    print(f"  → No features extracted for {fname}, skipping.")
            except Exception:
                print(f"  → Error extracting {fname}, skipping.")
                print(traceback.format_exc())
        else:
            print(f"  → File not found in tree: {fname}")

    if not all_features_list:
        print("No features extracted. Check your audioFolder and Extractor.")
        exit(1)

    new_df = pd.DataFrame(all_features_list)
    new_df.to_csv(va_file, index=False)
    print(f"Dataset built and saved to {va_file}")

else:
    print(f"Loading dataset from existing CSV: {va_file}")
    new_df = pd.read_csv(va_file)
    # coerce non-numeric to NaN, then drop
    for col in new_df.columns.drop(['file','value'], errors='ignore'):
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    new_df.dropna(inplace=True)

# ─── 2) PREPARE FEATURES & TARGET ────────────────────────────────────────────────
X = new_df.drop(columns=['file','value'], errors='ignore')
Y = new_df['value']

if X.shape[0] < 2:
    print("Not enough data to train. Exiting.")
    exit(1)

all_original_feature_names = X.columns.tolist()
print(f"Total features available: {len(all_original_feature_names)}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ─── 3) OPTUNA TUNING ───────────────────────────────────────────────────────────
def stop_if_good(study, trial):
    if trial.value is not None and trial.value >= 0.95:
        print(f"▶️ Stopping early: trial#{trial.number} reached {trial.value:.3f}")
        study.stop()

def lgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "num_leaves":       trial.suggest_int("num_leaves", 16, 128),
        "max_depth":        trial.suggest_int("max_depth", 3, 20),
        "learning_rate":    trial.suggest_loguniform("learning_rate", 1e-4, 1e-1),
        "subsample":        trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda":       trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "random_state":     42,
        "n_jobs":           -1
    }
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lgbm',   LGBMRegressor(**params))
    ])
    return cross_val_score(pipe, X_train, y_train, cv=3, scoring="r2").mean()

study = optuna.create_study(direction="maximize")
study.optimize(lgb_objective, n_trials=n_trials, callbacks=[stop_if_good])
best_params = study.best_params
print("▶️ Best LGBM params:", best_params)

# ─── 4) TRAIN FINAL PIPELINE ───────────────────────────────────────────────────
regr = Pipeline([
    ('scaler', StandardScaler()),
    ('lgbm',   LGBMRegressor(**best_params))
])
regr.fit(X_train, y_train)
print("Optuna-tuned LightGBM pipeline training complete.")

# ─── 5) EVALUATE & SAVE ─────────────────────────────────────────────────────────
y_pred = regr.predict(X_test)
print("\n--- Model Evaluation (LightGBM) ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
print(f"MAE:       {mean_absolute_error(y_test, y_pred):.3f}")
print(f"MSE:       {mean_squared_error(y_test, y_pred):.3f}")

# ensure parent dir exists
model_dir = os.path.dirname(model_output_name)
os.makedirs(model_dir, exist_ok=True)

joblib.dump({
    'model': regr,
    'all_original_features': all_original_feature_names
}, model_output_name)
print(f"Model and features saved to '{model_output_name}'")

# ─── 6) QUICK SCATTER PLOT ───────────────────────────────────────────────────────
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"True vs Predicted (LGBM) R²={r2_score(y_test, y_pred):.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()

# ─── 7) FEATURE IMPORTANCE (GAIN) ─────────────────────────────────────────────────
lgbm_model = regr.named_steps['lgbm']
gain_vals  = lgbm_model.booster_.feature_importance(importance_type='gain')
importances = pd.Series(gain_vals, index=all_original_feature_names).sort_values(ascending=False)

# print top N
top_n = 10
print(f"\nTop {top_n} features by gain:")
print(importances.head(top_n))

# bar‐plot of top N
plt.figure(figsize=(10,6))
importances.head(top_n).plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Gain")
plt.title(f"Top {top_n} Gain Importances")
plt.tight_layout()
plt.show()
