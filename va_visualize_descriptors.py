# main.py
import pandas as pd
import os
import numpy as np # Only needed for dummy data creation if you move it here
from sklearn.svm import SVR # Example for placeholder, remove if not needed in main
from sklearn.model_selection import train_test_split # Example for placeholder

# Import functions from your new modules
from va_data_loader import load_vad_lexicon, load_descriptor_pairs, load_input_va_points
from va_analyzer import perform_descriptor_analysis
from va_plotter import plot_va_space, plot_va_space_radial

# --- File Paths Configuration ---
VAD_LEXICON_DIR = "./datasets/NRC-VAD-Lexicon-v2.1/OneFilePerDimension"
AROUSAL_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "arousal-NRC-VAD-Lexicon-v2.1.txt")
VALENCE_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "valence-NRC-VAD-Lexicon-v2.1.txt")
DESCRIPTOR_PAIRS_PATH = "./datasets/descriptorPairs.txt"
INPUT_VALENCE_PATH = "./datasets/out_Valence.csv"
INPUT_AROUSAL_PATH = "./datasets/out_Arousal.csv"
ANALYSIS_RESULTS_OUTPUT_PATH = "va_point_descriptor_scores.csv"

if __name__ == "__main__":
    #create_dummy_files() # Ensure dummy files are available for execution
    
    print("--- Starting VA Descriptor Analysis ---")

    # 1. Load Lexicon Data
    arousal_lexicon = load_vad_lexicon(AROUSAL_LEXICON_PATH)
    valence_lexicon = load_vad_lexicon(VALENCE_LEXICON_PATH)

    if arousal_lexicon is None or valence_lexicon is None:
        print("Cannot proceed without VAD lexicon data. Exiting.")
        exit()

    # 2. Load Descriptor Pairs
    descriptor_pairs = load_descriptor_pairs(DESCRIPTOR_PAIRS_PATH, arousal_lexicon, valence_lexicon)
    if not descriptor_pairs:
        print("No descriptor pairs to analyze. Analysis results will be empty.")
        # Proceed to plotting if input VA points exist, but analysis won't run.
    
    # 3. Load Input VA Points (e.g., from predictions)
    input_va_points_df = load_input_va_points(INPUT_VALENCE_PATH, INPUT_AROUSAL_PATH)
    if input_va_points_df is None or input_va_points_df.empty:
        print("No input VA points found for analysis or plotting. Exiting.")
        exit()

    print("\nInput VA values (mean and std):")
    print(f"  Valence: Mean = {input_va_points_df['valence'].mean():.4f}, Std = {input_va_points_df['valence'].std():.4f}")
    print(f"  Arousal: Mean = {input_va_points_df['arousal'].mean():.4f}, Std = {input_va_points_df['arousal'].std():.4f}")
    print("-" * 50)

    # 4. Perform Descriptor Value Calculation
    analysis_results_df = perform_descriptor_analysis(input_va_points_df, descriptor_pairs)
    
    if not analysis_results_df.empty:
        print("\n--- Descriptor Value Analysis Results (first 5 rows) ---")
        print(analysis_results_df.head())
        analysis_results_df.to_csv(ANALYSIS_RESULTS_OUTPUT_PATH, index=False)
        print(f"\nDetailed scores saved to '{ANALYSIS_RESULTS_OUTPUT_PATH}'")
    else:
        print("\nNo descriptor analysis results to display.")


    # 5. Plotting
    plot_va_space_radial(input_va_points_df, descriptor_pairs)
    print("--- Program Finished ---")