import os
import pandas as pd
import joblib
import numpy as np

# Make sure these are accessible (e.g., in the same directory or correctly imported)
from extractor import Extractor # Your feature extractor

# --- Configuration ---
# Path to your models directory (if you store them in a subfolder)
MODELS_DIR = './trained_models/' # Assuming models are in the current directory

# Feature extraction parameters (MUST be the same as during training)
FRAME_SIZE = 2048
HOP_SIZE = 1024
SAMPLE_RATE = 32000

def get_model_filename(descriptor_name):
    """Generates the expected filename for a descriptor's trained model."""
    # This must match the saving convention in train_descriptor_model.py
    return f"trained_descriptor_{descriptor_name.replace(' ', '_').replace('/', '_')}_model.joblib"

def get_descriptor_name_from_model_filename(model_filename):
    """
    Extracts the descriptor name from the model's filename, reversing the
    naming convention used in train_descriptor_model.py.
    e.g., 'trained_descriptor_happy_sad_model.joblib' -> 'happy/sad'
    """
    if model_filename.startswith('trained_descriptor_') and model_filename.endswith('_model.joblib'):
        descriptor_name_raw = model_filename.replace('trained_descriptor_', '').replace('_model.joblib', '')
        return descriptor_name_raw.replace('_', '/') # Revert underscores to slashes if applicable
    return None

if __name__ == "__main__":
    print("--- Batch Descriptor Value Prediction Program ---")

    # --- User Input ---
    descriptor_input = input("Enter the filename of the trained descriptor model (e.g., happy, chaotic): ").strip()
    audio_directory_path = input("Enter the path to the directory containing WAV audio files: ").strip().replace("'", '')

    model_filename = get_model_filename(descriptor_input)
    model_full_path = os.path.join(MODELS_DIR, model_filename)
    print(model_full_path)

    if not os.path.exists(model_full_path):
        print(f"Error: Model file not found at '{model_full_path}'. Please check the path and filename.")
        exit()

    if not os.path.isdir(audio_directory_path):
        print(f"Error: Directory not found at '{audio_directory_path}'. Please check the path.")
        exit()

    # --- 1. Load Trained Descriptor Model ---
    descriptor_name = descriptor_input
    if not descriptor_name:
        print(f"Error: Could not parse descriptor name from model filename '{model_file_name}'.")
        print("Please ensure the model filename follows the 'trained_descriptor_NAME_model.joblib' convention.")
        exit()

    try:
        loaded_data = joblib.load(model_full_path)
        descriptor_model_pipeline = loaded_data['model']
        all_original_features_list = loaded_data['all_original_features']
        print(f"\nLoaded model for descriptor '{descriptor_name}' from '{model_full_path}'.")
        print(f"Model expects {len(all_original_features_list)} original features.")
    except FileNotFoundError: # This check is already done above, but good for redundancy or if MODELS_DIR changes
        print(f"Error: Model for '{descriptor_name}' not found at '{model_full_path}'.")
        print("Please ensure you have trained and saved this descriptor model using 'train_descriptor_model.py'.")
        exit()
    except KeyError as e:
        print(f"Error: Saved model file for '{descriptor_name}' is missing expected data (KeyError: {e}).")
        print("Please ensure it was trained and saved with the latest 'train_descriptor_model.py' version (which includes 'all_original_features').")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the model: {e}")
        exit()

    # --- 2. Initialize Feature Extractor ---
    s1 = Extractor(SAMPLE_RATE, FRAME_SIZE, HOP_SIZE)
    print(f"\nInitializing feature extractor with SR={SAMPLE_RATE}, FS={FRAME_SIZE}, HS={HOP_SIZE}.")

    # --- 3. Process Audio Files in Directory ---
    predictions_data = [] # To store {'filename': ..., 'predicted_value': ...}
    audio_files_found = [f for f in os.listdir(audio_directory_path) if f.lower().endswith('.wav')]

    if not audio_files_found:
        print(f"\nNo WAV files found in '{audio_directory_path}'. Exiting.")
        exit()

    print(f"\nFound {len(audio_files_found)} WAV files. Processing...")

    for i, audio_file_name in enumerate(audio_files_found):
        full_audio_path = os.path.join(audio_directory_path, audio_file_name)
        print(f"  ({i+1}/{len(audio_files_found)}) Processing: {audio_file_name}")

        try:
            # Extract features
            extracted_features_dict = s1.extract(full_audio_path)
            if not extracted_features_dict:
                print(f"    Warning: No features extracted from '{audio_audio_file_name}'. Skipping.")
                continue

            # Prepare features for prediction
            extracted_features_df = pd.DataFrame([extracted_features_dict]) 
            features_for_prediction = extracted_features_df.reindex(columns=all_original_features_list, fill_value=0.0)
            features_for_prediction = features_for_prediction.fillna(0.0) # Final NaN check

            # Make prediction
            predicted_value = descriptor_model_pipeline.predict(features_for_prediction)[0]
            predictions_data.append({
                'filename': audio_file_name,
                f'predicted_{descriptor_name.replace("/", "_")}': predicted_value
            })
            print(f"    Predicted value: {predicted_value:.4f}")

        except Exception as e:
            print(f"    Error processing '{audio_file_name}': {e}. Skipping this file.")

    # --- 4. Sort and Display Results ---
    if not predictions_data:
        print("\nNo successful predictions made. Check for errors during processing.")
    else:
        results_df = pd.DataFrame(predictions_data)
        # Sort by the predicted value column in descending order
        predicted_column_name = f'predicted_{descriptor_name.replace("/", "_")}'
        results_df_sorted = results_df.sort_values(by=predicted_column_name, ascending=False).reset_index(drop=True)

        print(f"\n--- Predicted '{descriptor_name}' Values (Decreasing Order) ---")
        print(results_df_sorted.to_string(index=False)) # Use to_string to avoid truncation

    print("\n--- Batch Prediction Complete ---")