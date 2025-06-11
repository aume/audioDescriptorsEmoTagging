import sqlite3
import math
import os
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from extractor import Extractor

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib # Import joblib for saving/loading models
import traceback # Import traceback for detailed error logging


# 1. fetch sounds from directory
# fetch rating value from database file
# 2. extract features from sound file
# df = [values,..][[features],..]
# 3. train and evaluate model 


build_dataset = True
#
# 1.
#
# !!! IMPORTANT: Set these paths based on whether you are training Valence or Arousal !!!
# Uncomment one block at a time to train each model:

# For Valence training:
# va_file = './datasets/out_Valence.csv'
# df_ratings_path = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Ratings/Valence.csv"
# model_output_name = "./trained_models/trained_valence_model.joblib"
# print("--- Configuring for Valence Model Training ---")


# For Arousal training:
va_file = './datasets/out_Arousal.csv'
df_ratings_path = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Ratings/Arousal.csv"
model_output_name = "./trained_models/trained_arousal_model.joblib"
print("--- Configuring for Arousal Model Training ---")


audioFolder = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Audio/600_Sounds/All/"


# 2.
#
if build_dataset:
    print('Building dataset by extracting features...')
    frame_size = 2048
    hop_size = 1024
    sample_rate = 32000

    s1 = Extractor(sample_rate, frame_size, hop_size)
    
    all_features_list = []
    
    df_ratings = pd.read_csv(df_ratings_path) # Load df_ratings here for consistency
    for index, row in df_ratings.iterrows():
        file_name = row[0]
        value = row[1]
        file_path = os.path.join(audioFolder, file_name)

        if os.path.exists(file_path):
            print(f"Extracting from: {file_name}")
            try:
                features_dict = s1.extract(file_path) # Returns a dict of features for one file
                if features_dict: # Only add if features were successfully extracted
                    features_dict['file'] = file_name
                    features_dict['value'] = float(value)
                    all_features_list.append(features_dict)
                else:
                    print(f"No features extracted for {file_name}. Skipping.")
            except Exception as e:
                print(f"Skipping {file_name}. An error occurred during feature extraction.")
                print(f"Error: {e}")
                print("--- Full Traceback ---")
                print(traceback.format_exc())
                print("----------------------")
        else:
            print(f"File does not exist: {file_path}. Skipping.")
    
    if not all_features_list:
        print("No features extracted. Ensure audio files exist and Extractor works correctly.")
        exit()

    new_df = pd.DataFrame(all_features_list)
    new_df.to_csv(va_file, index=False) 
    print(f"Dataset built and saved to {va_file}")
else:
    print(f"Loading dataset from existing CSV: {va_file}")
    new_df = pd.read_csv(va_file)
    # Ensure all columns are numeric, coerce errors to NaN and drop if any remain
    numeric_cols = new_df.columns.drop(['file', 'value'], errors='ignore')
    for col in numeric_cols:
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
    new_df.dropna(inplace=True) # Drop rows with any NaN after coercion


#
# 3.
#


X = new_df.drop(columns=['file', 'value'], errors='ignore') # Features
Y = new_df['value'] # Target VA value


# Check if there's enough data after feature extraction and cleaning
if X.empty or Y.empty or len(X) < 2:
    print("Not enough data to train the model after feature extraction/loading and cleaning. Exiting.")
    exit()


# --- NEW: Get the list of all original feature names before splitting and pipeline ---
all_original_feature_names = X.columns.tolist()
print(f"Total original features extracted and available for pipeline: {len(all_original_feature_names)}")


# feture selection
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# Ensure k is not greater than the number of available features
k_features = min(150, X.shape[1]) 
regr = Pipeline([ # CHANGED: Use Pipeline directly
    ('scaler', StandardScaler()),              # Name the scaler 'scaler'
    ('selector', SelectKBest(f_regression, k=k_features)), # Name the selector 'selector'
    ('svr', SVR(C=0.4, epsilon=0.01))          # Name the SVR 'svr'
])

regr.fit(X_train, y_train)
print("\nModel training complete.")


# Access the StandardScaler step by its name ('scaler')
scaler_step = regr.named_steps['scaler']

print("StandardScaler Parameters:")
print(f"  Mean (per feature): {scaler_step.mean_[:5]}... (first 5 values)") # Print first 5 for brevity
print(f"  Standard Deviation (per feature): {scaler_step.scale_[:5]}... (first 5 values)") # Print first 5 for brevity
print("-" * 50)

# Access the SelectKBest step by its name ('selector')
selector_step = regr.named_steps['selector']

# Get the boolean mask of selected features
selected_features_mask = selector_step.get_support()

# Get the names of the features that were input to SelectKBest
# This will be the columns of X_train (which is a subset of all_original_feature_names)
features_before_selection = X_train.columns

# Use the boolean mask to filter the original feature names
selected_feature_names = features_before_selection[selected_features_mask].tolist()

print(f"Selected Features by SelectKBest (k={selector_step.k}):")
print(selected_feature_names[:10]) # Print first 10 selected features
print(f"Total selected features: {len(selected_feature_names)}")

# Optional: Print scores of the selected features (useful for understanding why they were chosen)
print("\nScores for Selected Features:")
# selector_step.scores_ holds the scores for all features *before* selection
# We filter these scores using the same mask
selected_scores = selector_step.scores_[selected_features_mask]
selected_pvalues = selector_step.pvalues_[selected_features_mask] # if using f_classif/f_regression

for name, score, p_value in zip(selected_feature_names, selected_scores, selected_pvalues):
    print(f"  {name}: Score = {score:.4f}, P-value = {p_value:.4f}")


y_pred = regr.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"R2 Score: {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print("-" * 50)


# --- NEW: Save the model along with selected features and ALL original features ---
model_and_features_to_save = {
    'model': regr,
    'selected_features': selected_feature_names, # The 150 features used by SVR
    'all_original_features': all_original_feature_names # The full list of 172 features
}

joblib.dump(model_and_features_to_save, model_output_name)
print(f"\nModel and selected feature list saved to '{model_output_name}'")


# --- Plotting (using the input data for this specific run) ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title(f"True Values vs. Predicted Values (SVR) R2={r2:.3f}")
plt.grid(True)
plt.show()

print("\n--- Program Finished ---")