import os
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score # Only R2 is needed for visualization

# --- Configuration ---
DATABASE_NAME = './datasets/audio_predictions.db'
MODEL_PATH = './trained_models/'
# Random state used for splitting data during descriptor model training
DESCRIPTOR_MODEL_RANDOM_STATE = 42 

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
        
        features_wide_df = features_wide_df.fillna(0) # Fill NaNs if features are missing for some files

        # 4. Merge the wide features DataFrame with the descriptor scores DataFrame
        merged_df = pd.merge(features_wide_df, descriptor_df, on='filename', how='inner')

        if merged_df.empty:
            print(f"  Warning: No common audio files found between features and descriptor scores for '{selected_descriptor}'.")
            return None, None

        # Separate features (X) and target (Y)
        X = merged_df.drop(columns=['filename', 'score']) 
        Y = merged_df['score']

        X = X.apply(pd.to_numeric, errors='coerce') # Ensure all feature columns are numeric
        X = X.fillna(0) # Fill any NaNs that might arise from coercion

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

def train_descriptor_model_and_get_r2(X, Y, descriptor_name):
    """
    Trains an SVR model for the given descriptor and returns its R2 score.
    Does NOT save the model or plot individually.
    """
    if X.empty or Y.empty or len(X) < 2:
        print(f"  Skipping '{descriptor_name}': Not enough data to train the model. Need at least 2 samples.")
        return None

    # Split data using the consistent random_state
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=DESCRIPTOR_MODEL_RANDOM_STATE)

    # Determine k for SelectKBest (max 150 features, or less if fewer are available)
    k_features = min(150, X.shape[1]) 
    
    # Define the SVR pipeline
    regr_pipeline = Pipeline([
        ('scaler', StandardScaler()),              
        ('selector', SelectKBest(f_regression, k=k_features)), 
        ('svr', SVR(C=0.4, epsilon=0.001))          
    ])

    try:
        regr_pipeline.fit(X_train, y_train)
        y_pred = regr_pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return r2

    except Exception as e:
        print(f"  Error training or evaluating model for '{descriptor_name}': {e}")
        return None

if __name__ == "__main__":
    print("--- Automated Descriptor Model Training and R2 Visualization ---")

    # Check if the database exists
    if not os.path.exists(DATABASE_NAME):
        print(f"Error: Database '{DATABASE_NAME}' not found.")
        print("Please ensure you have run 'log_predictions_to_db.py' first to create and populate the database.")
        exit()

    # Get the list of all available descriptors
    descriptors = get_available_descriptors(DATABASE_NAME)

    if not descriptors:
        print("No descriptors found in the database. Please ensure the 'descriptor_scores' table is populated.")
        print("This typically happens after running 'predict_new_audio.py' and then 'log_predictions_to_db.py'.")
        exit()

    print(f"\nFound {len(descriptors)} unique descriptors. Starting training process...")

    all_r2_scores = {} # Dictionary to store R2 scores for all trained models

    # Loop through each descriptor, train a model, and collect its R2 score
    for i, descriptor in enumerate(descriptors):
        print(f"\n({i+1}/{len(descriptors)}) Training model for descriptor: '{descriptor}'")
        
        # Load data for the current descriptor
        X_data, Y_data = load_data_for_descriptor(DATABASE_NAME, descriptor)

        if X_data is not None and Y_data is not None:
            # Train model and get R2 score
            r2 = train_descriptor_model_and_get_r2(X_data, Y_data, descriptor)
            if r2 is not None:
                all_r2_scores[descriptor] = r2
                print(f"  R2 score for '{descriptor}': {r2:.3f}")
            else:
                print(f"  Failed to get R2 score for '{descriptor}'.")
        else:
            print(f"  Skipping '{descriptor}' due to data loading issues.")

    print("\n--- All descriptor models processed. Generating visualization ---")

    # Visualize all collected R2 Scores
    if not all_r2_scores:
        print("No R2 scores collected to visualize. Check for errors during training.")
    else:
        model_names = list(all_r2_scores.keys())
        r2_values = list(all_r2_scores.values())

        # Sort descriptors by R2 value for better readability in the plot
        sorted_indices = sorted(range(len(r2_values)), key=lambda k: r2_values[k], reverse=True)
        sorted_model_names = [model_names[i] for i in sorted_indices]
        sorted_r2_values = [r2_values[i] for i in sorted_indices]

        plt.figure(figsize=(12, max(6, len(sorted_model_names) * 0.4))) # Adjust fig size dynamically
        bars = plt.barh(sorted_model_names, sorted_r2_values, color='skyblue')
        plt.xlabel('R2 Score')
        plt.title('R2 Scores for All Descriptor Emotion Prediction Models')
        plt.xlim(0, 1.0) # R2 scores are typically between 0 and 1

        # Add R2 values as text labels on the bars
        for bar in bars:
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{bar.get_width():.3f}', va='center')

        plt.gca().invert_yaxis() # Put highest R2 at the top
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        
        plot_filename = "all_descriptors_r2_scores_plot.png"
        plt.savefig(plot_filename)
        print(f"\nVisualization saved as '{plot_filename}'")
        # plt.show() # Uncomment this line if you want the plot to pop up immediately

    print("\n--- Program Finished ---")