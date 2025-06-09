import os
import pandas as pd
import joblib
import numpy as np

# Make sure these are accessible (e.g., in the same directory or correctly imported)
from extractor import Extractor # Your feature extractor

# --- Configuration ---
# Path to your models directory (if you store them in a subfolder)
MODELS_DIR = './trained_models' # Assuming models are in the current directory as per previous scripts

# Feature extraction parameters (MUST be the same as during training)
FRAME_SIZE = 2048
HOP_SIZE = 1024
SAMPLE_RATE = 32000

def get_model_filename(descriptor_name):
    """Generates the expected filename for a descriptor's trained model."""
    # This must match the saving convention in train_descriptor_model.py
    return f"trained_descriptor_{descriptor_name.replace(' ', '_').replace('/', '_')}_model.joblib"

if __name__ == "__main__":
    print("--- Descriptor Value Prediction Program ---")

    # --- User Input ---
    descriptor_input = input("Enter the descriptor name (e.g., 'score_happy', 'score_alarming'): ").strip()
    audio_file_path = input("Enter the full path to the audio file (e.g., /path/to/my_audio.wav): ").strip().replace("'", '')

    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at '{audio_file_path}'. Please check the path.")
        exit()

    # --- 1. Load Trained Descriptor Model ---
    model_filename = get_model_filename(descriptor_input)
    model_full_path = os.path.join(MODELS_DIR, model_filename)
    print(model_full_path)

    try:
        loaded_data = joblib.load(model_full_path)
        descriptor_model_pipeline = loaded_data['model']
        all_original_features_list = loaded_data['all_original_features']
        print(f"\nLoaded model for descriptor '{descriptor_input}' from '{model_full_path}'.")
        print(f"Model expects {len(all_original_features_list)} original features.")
    except FileNotFoundError:
        print(f"Error: Model for '{descriptor_input}' not found at '{model_full_path}'.")
        print("Please ensure you have trained and saved this descriptor model using 'train_descriptor_model.py'.")
        exit()
    except KeyError as e:
        print(f"Error: Saved model file for '{descriptor_input}' is missing expected data (KeyError: {e}).")
        print("Please ensure it was trained and saved with the latest 'train_descriptor_model.py' version.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading the model: {e}")
        exit()

    # --- 2. Extract Features from Input Audio File ---
    s1 = Extractor(SAMPLE_RATE, FRAME_SIZE, HOP_SIZE)
    print(f"\nExtracting features from '{os.path.basename(audio_file_path)}'...")
    try:
        extracted_features_dict = s1.extract(audio_file_path)
        if not extracted_features_dict:
            print(f"Error: No features extracted from '{audio_file_path}'. Check file integrity or Extractor.")
            exit()
        print(f"Successfully extracted {len(extracted_features_dict)} features.")
    except Exception as e:
        print(f"Error during feature extraction for '{os.path.basename(audio_file_path)}': {e}")
        exit()

    # --- 3. Prepare Features for Prediction ---
    # Create a DataFrame from the extracted features
    # Convert single dictionary to DataFrame row
    extracted_features_df = pd.DataFrame([extracted_features_dict]) 

    # Reindex the DataFrame to match the all_original_features_list
    # This ensures correct number and order of features, filling missing with 0.0
    features_for_prediction = extracted_features_df.reindex(columns=all_original_features_list, fill_value=0.0)
    features_for_prediction = features_for_prediction.fillna(0.0) # Final NaN check

    if not features_for_prediction.columns.equals(pd.Index(all_original_features_list)):
        print("Warning: Feature columns mismatch after preparation for prediction.")
        # This warning might indicate an issue with your Extractor's consistency

    # --- 4. Make Prediction ---
    print("\nMaking prediction...")
    try:
        predicted_value = descriptor_model_pipeline.predict(features_for_prediction)[0] # [0] because predict returns an array
        print(f"\nPredicted '{descriptor_input}' score for '{os.path.basename(audio_file_path)}': {predicted_value:.4f}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Ensure the feature extraction and model loading were successful and consistent.")

    print("\n--- Prediction Complete ---")