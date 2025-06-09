import sqlite3
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt # NEW: Import matplotlib for plotting

# Configuration
DATABASE_NAME = './datasets/audio_predictions.db'
MODEL_PATH = './trained_models/'
FIGURE_PATH = './figures/'
def get_available_descriptors(db_name):
    """
    Fetches a list of unique descriptor names from the 'descriptor_scores' table
    in the specified SQLite database.
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
    It joins the features and descriptor scores based on file_id and pivots
    the features table into a wide format suitable for scikit-learn.
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
            print(f"Error: No audio features found in the database. Ensure 'audio_features' table is populated.")
            return None, None
        if descriptor_df.empty:
            print(f"Error: No scores found for descriptor '{selected_descriptor}'. Please check if it's logged correctly.")
            return None, None

        # 3. Pivot audio features from long format to wide format (features as columns)
        # 'filename' becomes the index, 'feature_name' becomes columns, 'feature_value' are the values.
        features_wide_df = features_long_df.pivot_table(
            index='filename',
            columns='feature_name',
            values='feature_value'
        )
        
        # Handle potential NaNs introduced by pivot_table (if some files miss certain features)
        # Filling with 0 is a common strategy, but consider alternatives if appropriate for your data.
        features_wide_df = features_wide_df.fillna(0)

        # 4. Merge the wide features DataFrame with the descriptor scores DataFrame
        # Merge on 'filename' to ensure correct alignment of features (X) and target (Y)
        merged_df = pd.merge(features_wide_df, descriptor_df, on='filename', how='inner')

        if merged_df.empty:
            print(f"No common audio files found between features and descriptor scores for '{selected_descriptor}'.")
            return None, None

        # Separate features (X) and target (Y)
        # Drop the 'filename' column as it's no longer needed for training and the index holds file identity
        X = merged_df.drop(columns=['filename', 'score']) 
        Y = merged_df['score']

        # Ensure all feature columns are numeric (coercing errors to NaN and then filling)
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0) # Fill any NaNs that might arise from coercion

        print(f"Successfully loaded {len(X)} samples for '{selected_descriptor}' with {X.shape[1]} features.")
        return X, Y

    except sqlite3.Error as e:
        print(f"Database error when loading data: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None, None
    finally:
        if conn:
            conn.close()

def train_descriptor_model(X, Y, descriptor_name):
    """
    Trains and evaluates an SVR model for the given descriptor using the provided
    features (X) and scores (Y), and plots true vs. predicted values.
    """
    if X.empty or Y.empty or len(X) < 2:
        print("Not enough data to train the model. Need at least 2 samples.")
        return

    print(f"\n--- Training Model for Descriptor: '{descriptor_name}' ---")
    print(f"Total samples for training: {len(X)}")
    print(f"Number of features being used: {X.shape[1]}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define the number of features for SelectKBest. 
    # Use min(150, actual_feature_count) to prevent errors if fewer than 150 features are available.
    k_features = min(150, X.shape[1]) 
    
    # Define the SVR pipeline: StandardScaler -> SelectKBest -> SVR
    regr_pipeline = Pipeline([
        ('scaler', StandardScaler()),              # Scales features to zero mean and unit variance
        ('selector', SelectKBest(f_regression, k=k_features)), # Selects top K features based on F-regression score
        ('svr', SVR(C=0.4, epsilon=0.001))          # Support Vector Regressor with specified hyperparameters
    ])

    try:
        # Train the model
        regr_pipeline.fit(X_train, y_train)
        print("Model training complete.")

        # Evaluate the model on the test set
        y_pred = regr_pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print("\n--- Model Evaluation ---")
        print(f"R2 Score: {r2:.3f}")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print("-" * 50)

        # NEW: Plot True vs Predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs. True')
        
        # Add a perfect prediction line (y=x)
        # Determine the range for the y=x line based on min/max of both true and predicted values
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.title(f'True vs. Predicted Scores for "{descriptor_name}"')
        plt.xlabel(f'True {descriptor_name} Score')
        plt.ylabel(f'Predicted {descriptor_name} Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot with a filename that includes the descriptor name
        plot_filename = f"true_vs_predicted_{descriptor_name.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(FIGURE_PATH+plot_filename)
        print(f"True vs. Predicted plot saved as '{plot_filename}'")
        plt.show() # Uncomment this line if you want the plot to pop up on your screen

        # Save the trained model
        # Replace spaces and slashes in descriptor name for a valid filename
        model_filename = f"trained_descriptor_{descriptor_name.replace(' ', '_').replace('/', '_')}_model.joblib"
        joblib.dump(regr_pipeline, MODEL_PATH+model_filename)
        print(f"Trained model saved as '{model_filename}'")

    except Exception as e:
        print(f"An error occurred during model training, evaluation, or plotting: {e}")

if __name__ == "__main__":
    print(f"--- Descriptor-based SVR Model Trainer ---")

    # Check if the database exists before proceeding
    if not os.path.exists(DATABASE_NAME):
        print(f"Error: Database '{DATABASE_NAME}' not found.")
        print("Please ensure you have run 'log_predictions_to_db.py' first to create and populate the database.")
        exit()

    # Get the list of unique descriptors available in the database
    descriptors = get_available_descriptors(DATABASE_NAME)

    if not descriptors:
        print("No descriptors found in the database. Please ensure the 'descriptor_scores' table is populated.")
        print("This typically happens after running 'predict_new_audio.py' and then 'log_predictions_to_db.py'.")
        exit()

    print("\nAvailable Descriptors for Training:")
    for i, desc in enumerate(descriptors):
        print(f"{i+1}. {desc}")

    # Loop to allow training multiple descriptor models
    while True:
        try:
            choice = input("\nEnter the number or full name of the descriptor you want to train a model for (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print("Exiting program.")
                break

            selected_descriptor = None
            if choice.isdigit():
                # User entered a number, try to map it to a descriptor from the list
                idx = int(choice) - 1 # Adjust for 0-based indexing
                if 0 <= idx < len(descriptors):
                    selected_descriptor = descriptors[idx]
                else:
                    print("Invalid number. Please enter a number from the list.")
            else:
                # User entered a name, check if it's in the list of descriptors
                if choice in descriptors:
                    selected_descriptor = choice
                else:
                    print("Descriptor name not found. Please ensure correct spelling and case (it's case-sensitive).")
            
            if selected_descriptor:
                print(f"Attempting to load data for descriptor: '{selected_descriptor}'...")
                X_data, Y_data = load_data_for_descriptor(DATABASE_NAME, selected_descriptor)

                if X_data is not None and Y_data is not None:
                    train_descriptor_model(X_data, Y_data, selected_descriptor)
                
                # Ask the user if they want to train another model or quit
                another_round = input("\nTrain another descriptor model? (y/n): ").strip().lower()
                if another_round != 'y':
                    print("Exiting program.")
                    break # Exit the loop if user doesn't want to train more
                else:
                    print("\n--- Preparing to train another model ---")
                    # Loop will naturally prompt for next descriptor

        except ValueError:
            print("Invalid input format. Please enter a number or the exact descriptor name.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\n--- Program Finished ---")