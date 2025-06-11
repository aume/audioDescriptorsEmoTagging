import os
import pandas as pd
import joblib
import numpy as np

# Make sure these are accessible (e.g., in the same directory or correctly imported from a package)
from extractor import Extractor # feature extractor
from va_data_loader import load_vad_lexicon, load_descriptor_pairs
from va_analyzer import perform_descriptor_analysis

# --- Configuration ---
# Paths to  saved models 
VALENCE_MODEL_PATH = "./trained_models/trained_valence_model.joblib" # Updated name to match saving in main.py
AROUSAL_MODEL_PATH = "./trained_models/trained_arousal_model.joblib" # Update if you save arousal separately

# Path to the folder containing the new audio files you want to analyze
NEW_AUDIO_FOLDER = "./datasets/HLD_corpus/" # IMPORTANT: put  new .wav files here!

# Paths to VAD lexicon and descriptor pairs (same as in main.py)
VAD_LEXICON_DIR = "./datasets/NRC-VAD-Lexicon-v2.1/OneFilePerDimension"
AROUSAL_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "arousal-NRC-VAD-Lexicon-v2.1.txt")
VALENCE_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "valence-NRC-VAD-Lexicon-v2.1.txt")
DESCRIPTOR_PAIRS_PATH = "./datasets/descriptorPairs.txt"

DATASETS_DIR = "./datasets/"

# Feature extraction parameters (MUST be the same as during training)
FRAME_SIZE = 2048
HOP_SIZE = 1024
SAMPLE_RATE = 32000

# --- Main Prediction Logic ---
if __name__ == "__main__":
    print("--- Starting Prediction for New Audio Files ---")

    # 1. Load Trained Valence and Arousal Models AND their feature lists
    try:
        valence_data = joblib.load(VALENCE_MODEL_PATH)
        valence_model = valence_data['model']
        valence_selected_features_list = valence_data['selected_features'] # The 150 selected features
        valence_all_original_features_list = valence_data['all_original_features'] # The 172 original features (NEW)
        print(f"Loaded Valence model from {VALENCE_MODEL_PATH}.")
        print(f"  Model expects {len(valence_all_original_features_list)} original features and uses {len(valence_selected_features_list)} selected features.")

        arousal_data = joblib.load(AROUSAL_MODEL_PATH)
        arousal_model = arousal_data['model']
        arousal_selected_features_list = arousal_data['selected_features']
        arousal_all_original_features_list = arousal_data['all_original_features'] # The 172 original features (NEW)
        print(f"Loaded Arousal model from {AROUSAL_MODEL_PATH}.")
        print(f"  Model expects {len(arousal_all_original_features_list)} original features and uses {len(arousal_selected_features_list)} selected features.")

    except FileNotFoundError:
        print(f"Error: One or both models not found. Ensure '{VALENCE_MODEL_PATH}' and '{AROUSAL_MODEL_PATH}' exist.")
        print("Please train and save models using main.py first.")
        exit()
    except KeyError as e:
        print(f"Error: Saved model file does not contain expected key: {e}. Please retrain models with the updated main.py.")
        exit()

    # 2. Initialize Feature Extractor
    s1 = Extractor(SAMPLE_RATE, FRAME_SIZE, HOP_SIZE)
    print(f"\nInitialized Extractor with SR:{SAMPLE_RATE}, FS:{FRAME_SIZE}, HS:{HOP_SIZE}")

    # 3. Extract Features from New Audio Files
    new_audio_features_dicts = [] # List of dictionaries, each dict for one audio file's features
    new_audio_filenames = []

    print(f"\nExtracting features from audio files in: {NEW_AUDIO_FOLDER}")
    if not os.path.exists(NEW_AUDIO_FOLDER):
        print(f"Error: New audio folder '{NEW_AUDIO_FOLDER}' not found. Please create it and add .wav files.")
        exit()

    for filename in os.listdir(NEW_AUDIO_FOLDER):
        if filename.endswith(".wav"):
            file_path = os.path.join(NEW_AUDIO_FOLDER, filename)
            print(f"Processing: {filename}")
            try:
                features_dict = s1.extract(file_path) # Returns a dict of features
                if features_dict: # Ensure features were extracted successfully
                    new_audio_features_dicts.append(features_dict)
                    new_audio_filenames.append(filename)
                else:
                    print(f"No features extracted for {filename}. Skipping.")
            except Exception as e:
                print(f"Error extracting features from {filename}: {e}")
                continue

    if not new_audio_features_dicts:
        print("No audio files processed or no features extracted. Exiting.")
        exit()

    # Create a DataFrame from the extracted feature dictionaries
    new_features_raw_df = pd.DataFrame(new_audio_features_dicts)
    
    # Assign the collected filenames as a new 'file' column
    new_features_raw_df['file'] = new_audio_filenames
    
    new_features_raw_df.to_csv(DATASETS_DIR+"new_audio_features_raw.csv", index=False)
    print("Raw extracted features saved to 'new_audio_features_raw.csv'")
    
    print(f"Extracted features for {len(new_features_raw_df)} new audio files.")
    if not new_features_raw_df.empty:
        print(f"Raw extracted features columns: {new_features_raw_df.columns.tolist()[:5]}...") # Show first 5
        print(f"Total features extracted for new audio: {new_features_raw_df.shape[1]}")
    else:
        print("No features extracted into DataFrame.")


    # 4. Prepare New Features for Prediction (filter and reorder according to trained models' features)
    # It's crucial that the new features DataFrame has the exact same columns, in the exact same order
    # as the features the model was trained on.
    
    # --- IMPORTANT CHANGE HERE ---
    # Reindex against ALL original features from training, not just selected ones.
    # The pipeline's StandardScaler expects the full set of features it was trained on.
    X_new_valence = new_features_raw_df.reindex(columns=valence_all_original_features_list, fill_value=0.0)
    X_new_arousal = new_features_raw_df.reindex(columns=arousal_all_original_features_list, fill_value=0.0)

    # Ensure no NaNs remain after reindexing (they should be filled by fill_value, but a double check)
    X_new_valence = X_new_valence.fillna(0.0)
    X_new_arousal = X_new_arousal.fillna(0.0)

    # Validate that columns are now correctly aligned
    if not X_new_valence.columns.equals(pd.Index(valence_all_original_features_list)):
        print("Warning: Valence features column mismatch after reindexing (full set).")
    if not X_new_arousal.columns.equals(pd.Index(arousal_all_original_features_list)):
        print("Warning: Arousal features column mismatch after reindexing (full set).")

    print(f"Prepared Valence features shape for prediction: {X_new_valence.shape}")
    print(f"Prepared Arousal features shape for prediction: {X_new_arousal.shape}")

    # 5. Predict Valence and Arousal for new files
    print("\nPredicting Valence and Arousal for new audio files...")
    # These lines are where the error occurred, now with correctly shaped input
    predicted_valence = valence_model.predict(X_new_valence)
    predicted_arousal = arousal_model.predict(X_new_arousal)
    
    print("Prediction complete.")

    # 6. Prepare Predicted VA for Descriptor Analysis
    predicted_va_df = pd.DataFrame({
        'file': new_audio_filenames,
        'valence': predicted_valence,
        'arousal': predicted_arousal
    })
    predicted_va_df.to_csv(DATASETS_DIR+"new_audio_predicted_va.csv", index=False)
    print("Predicted VA scores saved to 'new_audio_predicted_va.csv'")
    print("\nPredicted VA Points (first 5):")
    print(predicted_va_df.head())

    # 7. Load VAD Lexicon and Descriptor Pairs (for descriptor analysis)
    arousal_lexicon = load_vad_lexicon(AROUSAL_LEXICON_PATH)
    valence_lexicon = load_vad_lexicon(VALENCE_LEXICON_PATH)
    if arousal_lexicon is None or valence_lexicon is None:
        print("Cannot load VAD lexicons. Exiting descriptor analysis.")
        exit()

    descriptor_pairs = load_descriptor_pairs(DESCRIPTOR_PAIRS_PATH, arousal_lexicon, valence_lexicon)
    if not descriptor_pairs:
        print("No descriptor pairs loaded. Descriptor analysis will be skipped.")

    # 8. Calculate Descriptor Values
    if not predicted_va_df.empty and descriptor_pairs:
        descriptor_results_df = perform_descriptor_analysis(predicted_va_df, descriptor_pairs)
        print("\n--- Descriptor Value Analysis Results for New Audio (first 5 rows) ---")
        print(descriptor_results_df.head())
        
        # Save results to CSV
        output_csv_path = "new_audio_descriptor_scores.csv"
        descriptor_results_df.to_csv(DATASETS_DIR+output_csv_path, index=False)
        print(f"\nDetailed descriptor scores saved to '{output_csv_path}'")
    else:
        print("Skipping descriptor analysis due to missing VA predictions or descriptor pairs.")

    print("\n--- Prediction Process Complete ---")