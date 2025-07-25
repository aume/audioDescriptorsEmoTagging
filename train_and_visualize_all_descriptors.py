import os
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# NEW: use LightGBM 
from lightgbm import LGBMRegressor
import optuna
import csv

# --- Configuration ---
DATABASE_NAME = './datasets/audio_predictions_lightgbm.db'
# Random state used for splitting data during descriptor model training
DESCRIPTOR_MODEL_RANDOM_STATE = 42
# Directory to save all trained descriptor models
MODELS_SAVE_DIR = './trained_models_lightgbm/hld_models/'

# NEW: how many Optuna trials to run
n_trials = 60
# NEW: subset of descriptors to train (empty list => all)
DESCRIPTORS_TO_RUN = []

def get_available_descriptors(db_name):
    """
    Fetches a list of unique descriptor names from the 'descriptor_scores' table.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT descriptor_name FROM descriptor_scores ORDER BY descriptor_name")
        descriptors = [row[0] for row in cursor.fetchall()]
        return descriptors
    except sqlite3.Error as e:
        print(f"Database error when fetching descriptors: {e}")
        return []
    finally:
        if conn:
            conn.close()

def load_data_for_descriptor(db_name, selected_descriptor):
    """
    Loads audio features and the scores for a specific descriptor from the database.
    Pivots features to a wide format (features as columns) suitable for scikit-learn.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        
        # 1. Load audio features in a long format, including filename for linking
        features_query = """
        SELECT
            af.filename,
            afeat.feature_name,
            afeat.feature_value
        FROM
            audio_features afeat
        JOIN
            audio_files af ON afeat.file_id = af.id
        """
        features_long_df = pd.read_sql_query(features_query, conn)

        # 2. Load descriptor scores for the selected descriptor, including filename
        descriptor_query = """
        SELECT
            af.filename,
            ds.score
        FROM
            descriptor_scores ds
        JOIN
            audio_files af ON ds.file_id = af.id
        WHERE
            ds.descriptor_name = ?
        """
        descriptor_df = pd.read_sql_query(descriptor_query, conn, params=(selected_descriptor,))

        if features_long_df.empty:
            print(f"  Warning: No audio features found in the database. Ensure 'audio_features' table is populated.")
            return None, None
        if descriptor_df.empty:
            print(f"  Warning: No scores found for descriptor '{selected_descriptor}'. Please check if it's logged correctly.")
            return None, None

        # 3. Pivot audio features from long format to wide format (features as columns)
        features_wide_df = features_long_df.pivot_table(
            index='filename',
            columns='feature_name',
            values='feature_value'
        )
        features_wide_df = features_wide_df.fillna(0)  # Fill NaNs if features are missing for some files

        # 4. Merge the wide features DataFrame with the descriptor scores DataFrame
        merged_df = pd.merge(features_wide_df, descriptor_df, on='filename', how='inner')

        if merged_df.empty:
            print(f"  Warning: No common audio files found between features and descriptor scores for '{selected_descriptor}'.")
            return None, None

        # Separate features (X) and target (Y)
        X = merged_df.drop(columns=['filename', 'score'])
        Y = merged_df['score']

        X = X.apply(pd.to_numeric, errors='coerce')  # Ensure all feature columns are numeric
        X = X.fillna(0)  # Fill any NaNs that might arise from coercion

        return X, Y

    except sqlite3.Error as e:
        print(f"  Database error when loading data for descriptor '{selected_descriptor}': {e}")
        return None, None
    except Exception as e:
        print(f"  An unexpected error occurred during data loading for descriptor '{selected_descriptor}': {e}")
        return None, None
    finally:
        if conn:
            conn.close()

# NEW: early stopping callback for Optuna when R2 ≥ 0.8
def stop_if_good(study, trial):
    if trial.value is not None and trial.value >= 0.7:
        print(f"▶️ Stopping early: trial#{trial.number} reached {trial.value:.3f}")
        study.stop()

def train_descriptor_model_and_get_r2(X, Y, descriptor_name, save_dir):
    """
    Trains an LightGBM model for the given descriptor, saves the model,
    and returns its R2 score.
    """

    os.makedirs(save_dir, exist_ok=True)

    if X.empty or Y.empty or len(X) < 2:
        print(f"  Skipping '{descriptor_name}': Not enough data to train the model. Need at least 2 samples.")
        return None

    # Store all original feature names for later use
    all_original_feature_names = X.columns.tolist()

    # Split data using the consistent random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=DESCRIPTOR_MODEL_RANDOM_STATE
    )

    # -------------------------------------------------------------------------
    # Optuna tuning for LightGBM hyperparameters
    # -------------------------------------------------------------------------
    def objective(trial):
        # NEW: LightGBM hyperparameters
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 50, 300),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 300),
            "max_depth":          trial.suggest_int("max_depth", 3, 15),
            "learning_rate":      trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
            "subsample":          trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha":          trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
            "reg_lambda":         trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
            "random_state":       DESCRIPTOR_MODEL_RANDOM_STATE,
            "n_jobs":             -1,
            #"verbose":            -1   # silence LightGBM logs
        }
        model = LGBMRegressor(**params)  # NEW: instantiate LGBMRegressor

        return cross_val_score(model, X_train, y_train, cv=3, scoring="r2").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_if_good])

    best_params = study.best_params
    print(f"  Best LightGBM params for '{descriptor_name}': {best_params}")

    # -------------------------------------------------------------------------
    # Retrain final pipeline with best parameters
    # -------------------------------------------------------------------------
    # NEW: final pipeline with LightGBM
    regr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lgbm", LGBMRegressor(**best_params))
    ])

    try:
        # Train the model
        regr_pipeline.fit(X_train, y_train)
        print("  Model training complete.")

        # Predict & evaluate
        y_pred = regr_pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"  R2 for '{descriptor_name}': {r2:.3f}, MAE: {mae:.3f}")

        # Plot true vs. predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs True')
        minv, maxv = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([minv, maxv], [minv, maxv], 'r--', label='Perfect Prediction')
        plt.title(f'"{descriptor_name}" LightGBM: True vs Predicted')
        plt.xlabel("True Score")
        plt.ylabel("Predicted Score")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_fn = f"true_vs_predicted_{descriptor_name.replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, plot_fn))
        print(f"  Plot saved as '{plot_fn}'")

        # Save model + feature list
        model_fn = f"trained_descriptor_{descriptor_name.replace(' ', '_')}_lightgbm.joblib"
        joblib.dump(
            {'model': regr_pipeline, 'features': all_original_feature_names},
            os.path.join(save_dir, model_fn)
        )
        print(f"  Saved LightGBM model as '{model_fn}'")

        # return the score, the fitted pipeline, and the feature names
        return r2, regr_pipeline, all_original_feature_names

    except Exception as e:
        print(f"  Error for '{descriptor_name}': {e}")
        return None

if __name__ == "__main__":
    print("--- Automated Descriptor Model Training and R2 Visualization ---")

    # Ensure the models save directory exists
    os.makedirs(MODELS_SAVE_DIR, exist_ok=True)
    print(f"Models will be saved in: '{MODELS_SAVE_DIR}'")

    # Check if the database exists
    if not os.path.exists(DATABASE_NAME):
        print(f"Error: Database '{DATABASE_NAME}' not found.")
        print("Please ensure you have run 'log_predictions_to_db.py' first to create and populate the database.")
        exit()

    # Get the list of all available descriptors
    all_descriptors = get_available_descriptors(DATABASE_NAME)
    # NEW: only run for the ones you list
    if DESCRIPTORS_TO_RUN:
        descriptors = [d for d in all_descriptors if d in DESCRIPTORS_TO_RUN]
    else:
        descriptors = all_descriptors

    print(f"\nRunning on {len(descriptors)} descriptors: {descriptors}")

    if not descriptors:
        print("No descriptors found in the database. Please ensure the 'descriptor_scores' table is populated.")
        exit()

    all_r2_scores = {}  # Dictionary to store R2 scores for all trained models
    all_importances = {}  # Dictionary to store feature importances scores for all trained models

    # Loop through each descriptor, train a model, and collect its R2 score
    for i, descriptor in enumerate(descriptors, 1):
        print(f"\n({i}/{len(descriptors)}) Training model for descriptor: '{descriptor}'")
        X_data, Y_data = load_data_for_descriptor(DATABASE_NAME, descriptor)

        if X_data is not None and Y_data is not None:

            # r2 = train_descriptor_model_and_get_r2(X_data, Y_data, descriptor, MODELS_SAVE_DIR)

            result = train_descriptor_model_and_get_r2(X_data, Y_data, descriptor, MODELS_SAVE_DIR)
            if result is not None:
                # now result is (r2, pipeline, feat_list)
                r2, regr_pipeline, all_original_feature_names = result

                all_r2_scores[descriptor] = r2

                # extract feature importances
                # imps = regr_pipeline.named_steps['lgbm'].feature_importances_
                # all_importances[descriptor] = pd.Series(imps, index=all_original_feature_names)

                # gain‐based importances via the underlying Booster
                lgbm      = regr_pipeline.named_steps['lgbm']
                gain_vals = lgbm.booster_.feature_importance(importance_type='gain')
                # use your saved feature list so names stay correct
                feat_names = all_original_feature_names
                all_importances[descriptor] = pd.Series(gain_vals, index=feat_names)


                print(f"  R2 score for '{descriptor}': {r2:.3f}")
            else:
                print(f"  Failed to get R2 score for '{descriptor}'.")
        else:
            print(f"  Skipping '{descriptor}' due to data loading issues.")
    
    # NEW: FEATURE IMPORTANCE
    # ─── right here: EXPORT TOP-10 FEATURES & SAVE BAR CHARTS ─────────────────
    TOP_N = 10
    OUT_DIR = os.path.join(MODELS_SAVE_DIR, 'features', 'hld')
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) dump top-10 to CSV
    csv_path = os.path.join(OUT_DIR, 'top_features_by_descriptor.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['descriptor'] + [f for i in range(1, TOP_N+1) for f in (f'feat_{i}', f'value_{i}')]
        writer.writerow(header)
        for desc, imps in all_importances.items():
            top = imps.nlargest(TOP_N)
            row = [desc] + [elem for pair in zip(top.index, top.values) for elem in (pair[0], f"{pair[1]:.6f}")]
            writer.writerow(row)
    print(f"Saved top-{TOP_N} features CSV → {csv_path}")

    # 2) save a bar chart per descriptor
    for desc, imps in all_importances.items():
        top = imps.nlargest(TOP_N)
        plt.figure(figsize=(8,4))
        top.plot(kind='bar')
        plt.title(f"Top {TOP_N} Features for '{desc}'")
        plt.ylabel("Importance")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        chart_fn = f"{desc.replace(' ', '_')}_top{TOP_N}_features.png"
        plt.savefig(os.path.join(OUT_DIR, chart_fn))
        plt.close()
    print(f"Saved per-descriptor feature charts to {OUT_DIR}")

    # ─── then continue with your existing grouped-R2 visualization ──────────────
    print("\n--- All descriptor models processed. Generating visualization ---")

    if not all_r2_scores:
        print("No R2 scores collected to visualize. Check for errors during training.")
    else:
        # NEW GROUPED VISUALIZATION CODE
        # ─── Load descriptor pairs from the text file ──────────────────────────────
        pair_file = './datasets/descriptorPairs.txt'
        descriptor_pairs = []
        try:
            with open(pair_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('/')
                    if len(parts) == 2:
                        descriptor_pairs.append((parts[0].strip(), parts[1].strip()))
        except FileNotFoundError:
            print(f"Error: descriptor pairs file '{pair_file}' not found.")
            exit()

        valid_pairs = [pair for pair in descriptor_pairs
                    if any(d in all_r2_scores for d in pair)]

        pair_scores = []
        for pair in valid_pairs:
            top_score = max(all_r2_scores.get(d, -1) for d in pair)
            pair_scores.append((pair, top_score))
        pair_scores.sort(key=lambda x: x[1], reverse=True)

        colors = ['blue', 'orange']
        ordered_names, ordered_r2, bar_colors = [], [], []

        for idx, (pair, _) in enumerate(pair_scores):
            present = sorted([d for d in pair if d in all_r2_scores],
                            key=lambda d: all_r2_scores[d], reverse=True)
            for d in present:
                ordered_names.append(d)
                ordered_r2.append(all_r2_scores[d])
                bar_colors.append(colors[idx % 2])

        plt.figure(figsize=(12, max(6, len(ordered_names)*0.4)))
        y_pos = range(len(ordered_names))
        bars  = plt.barh(y_pos, ordered_r2, color=bar_colors)
        plt.yticks(y_pos, ordered_names)
        plt.xlabel('R2 Score')
        plt.title('R2 Scores Grouped by Descriptor Pair (LightGBM)')
        plt.xlim(0, 1.0)

        for bar in bars:
            w = bar.get_width()
            plt.text(w+0.01, bar.get_y()+bar.get_height()/2,
                    f'{w:.3f}', va='center')

        plt.gca().invert_yaxis()
        plt.tight_layout()

        plot_filename = "lightgbm_r2_hld.png"
        plt.savefig(plot_filename)
        print(f"\nGrouped visualization saved as '{plot_filename}'")

    print("\n--- Program Finished ---")
