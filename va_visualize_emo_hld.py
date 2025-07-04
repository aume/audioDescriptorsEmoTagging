import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from emo_file_mapping import build_file_mapping
from va_data_loader import load_vad_lexicon, load_descriptor_pairs

def plot_hld_in_va_space(hld_name, file_hld_path, file_va_path, ss_corpus_path, descriptor_pair, ax=None):
    """
    Loads data and draws a scatter plot on a given matplotlib axis (ax).
    If ax is None, it creates a new figure for a single plot.

    Args:
        hld_name (str): The name of the HLD to plot (matching filename case).
        ... (other args) ...
        ax (matplotlib.axes.Axes, optional): The subplot to draw on. Defaults to None.
    """
    # --- 1. Data Loading and Merging (condensed for brevity) ---
    number_to_name_map = build_file_mapping(ss_corpus_path)
    if not number_to_name_map: return
    name_to_number_map = {v: k for k, v in number_to_name_map.items()}
    
    try:
        arousal_df = pd.read_csv(os.path.join(file_va_path, 'arousal.csv'), header=None, names=['audio_file_name_wav', 'arousal'])
        valence_df = pd.read_csv(os.path.join(file_va_path, 'valence.csv'), header=None, names=['audio_file_name_wav', 'valence'])
        hld_df = pd.read_csv(os.path.join(file_hld_path, f'{hld_name}.csv'))
        hld_df.rename(columns={'File': 'file_number', 'Value': hld_name}, inplace=True)
    except FileNotFoundError as e:
        print(f"Error loading data for {hld_name}: {e}.")
        return

    arousal_df['audio file name'] = arousal_df['audio_file_name_wav'].str.replace('.wav', '', regex=False)
    valence_df['audio file name'] = valence_df['audio_file_name_wav'].str.replace('.wav', '', regex=False)
    arousal_df['file_number'] = arousal_df['audio file name'].map(name_to_number_map)
    valence_df['file_number'] = valence_df['audio file name'].map(name_to_number_map)

    arousal_df.dropna(subset=['file_number'], inplace=True); valence_df.dropna(subset=['file_number'], inplace=True)
    arousal_df['file_number'] = arousal_df['file_number'].astype(int); valence_df['file_number'] = valence_df['file_number'].astype(int)
    hld_df['file_number'] = hld_df['file_number'].astype(int)

    va_df = pd.merge(arousal_df[['file_number', 'arousal']], valence_df[['file_number', 'valence']], on='file_number')
    merged_df = pd.merge(va_df, hld_df, on='file_number')

    if merged_df.empty:
        print(f"No data to plot for {hld_name}.")
        if ax: ax.set_visible(False)
        return
        
    # --- 2. Plotting Logic ---
    show_plot_when_done = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        show_plot_when_done = True

    # Get original descriptors and their coordinates from the pair
    lexicon_d1, lexicon_d2 = descriptor_pair['d1'], descriptor_pair['d2']
    d1_coord, d2_coord = descriptor_pair['d1_coord'], descriptor_pair['d2_coord']

    # FIX: Determine the primary and secondary descriptor for consistent titling.
    # The primary descriptor is the one whose data is being plotted (hld_name).
    if hld_name.lower() == lexicon_d1:
        primary_desc = lexicon_d1
        secondary_desc = lexicon_d2
    else:
        primary_desc = lexicon_d2
        secondary_desc = lexicon_d1

    # Use a bipolar colormap like 'coolwarm'
    scatter = ax.scatter(
        merged_df['valence'], merged_df['arousal'], c=merged_df[hld_name], 
        cmap='coolwarm', alpha=0.7, zorder=2
    )
    
    # Add a colorbar
    fig = plt.gcf()
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{hld_name.capitalize()} Score')

    # Plot and annotate the descriptor points. The labels must match their original coordinates.
    ax.scatter(d1_coord[0], d1_coord[1], color='black', marker='*', s=200, zorder=5)
    ax.scatter(d2_coord[0], d2_coord[1], color='black', marker='*', s=200, zorder=5)
    ax.annotate(lexicon_d1, (d1_coord[0], d1_coord[1]), textcoords="offset points", xytext=(0,15), ha='center', weight='bold')
    ax.annotate(lexicon_d2, (d2_coord[0], d2_coord[1]), textcoords="offset points", xytext=(0,15), ha='center', weight='bold')

    # Set the title using the consistent primary/secondary order
    ax.set_title(f'"{primary_desc.capitalize()}/{secondary_desc.capitalize()}" Pair', fontsize=12)
    ax.set_xlabel('Valence'); ax.set_ylabel('Arousal')

    x_min = min(merged_df['valence'].min(), d1_coord[0], d2_coord[0]); x_max = max(merged_df['valence'].max(), d1_coord[0], d2_coord[0])
    y_min = min(merged_df['arousal'].min(), d1_coord[1], d2_coord[1]); y_max = max(merged_df['arousal'].max(), d1_coord[1], d2_coord[1])
    ax.set_xlim(x_min - 0.1, x_max + 0.1); ax.set_ylim(y_min - 0.1, y_max + 0.1)
    
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, zorder=1)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8, zorder=1)
    
    if show_plot_when_done:
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # --- CONFIGURATION ---
    TILE_PLOTS = True # Set to True to tile all plots, False to show one by one.
    
    # --- PATHS ---
    FILE_HLD_PATH = "./datasets/EmoSS_Ratings"
    FILE_VA_PATH = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Ratings"
    SS_CORPUS_PATH = "./datasets/EmoSS_Ratings/ES_Corpus"
    VAD_LEXICON_DIR = "./datasets/NRC-VAD-Lexicon-v2.1/OneFilePerDimension"
    AROUSAL_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "arousal-NRC-VAD-Lexicon-v2.1.txt")
    VALENCE_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "valence-NRC-VAD-Lexicon-v2.1.txt")
    DESCRIPTOR_PAIRS_PATH = "./datasets/descriptorPairs.txt"

    # --- Pre-computation ---
    print("Loading lexicon and descriptor data...")
    arousal_lexicon = load_vad_lexicon(AROUSAL_LEXICON_PATH)
    valence_lexicon = load_vad_lexicon(VALENCE_LEXICON_PATH)
    if arousal_lexicon is None or valence_lexicon is None: exit("Cannot proceed without VAD lexicon data.")
    descriptor_pairs = load_descriptor_pairs(DESCRIPTOR_PAIRS_PATH, arousal_lexicon, valence_lexicon)
    if not descriptor_pairs: exit("No descriptor pairs were loaded.")
        
    # --- Find all valid HLD files to plot ---
    valid_plots_info = []
    print(f"\nSearching for HLD files in: {FILE_HLD_PATH}")
    for filename in os.listdir(FILE_HLD_PATH):
        if filename.lower().endswith('.csv'):
            hld_name_original_case = os.path.splitext(filename)[0]
            hld_name_lower = hld_name_original_case.lower()
            
            target_pair = next((p for p in descriptor_pairs if p['d1'] == hld_name_lower or p['d2'] == hld_name_lower), None)
            
            if target_pair:
                valid_plots_info.append({'hld_name': hld_name_original_case, 'pair': target_pair})
            else:
                print(f"Warning: Could not find a matching descriptor pair for '{hld_name_lower}'. Skipping.")

    if not valid_plots_info:
        exit("No valid HLDs with matching descriptor pairs found.")

    # --- PLOTTING ---
    if TILE_PLOTS:
        num_plots = len(valid_plots_info)
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        fig.suptitle("High-Level Descriptors in Valence-Arousal Space", fontsize=16, weight='bold')
        axes = axes.flatten()

        for i, plot_info in enumerate(valid_plots_info):
            print(f"\n--- Tiling plot {i+1}/{num_plots}: {plot_info['hld_name']} ---")
            plot_hld_in_va_space(
                hld_name=plot_info['hld_name'],
                file_hld_path=FILE_HLD_PATH,
                file_va_path=FILE_VA_PATH,
                ss_corpus_path=SS_CORPUS_PATH,
                descriptor_pair=plot_info['pair'],
                ax=axes[i]
            )
        
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
            
        fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3)
        plt.show()

    else: # Show plots one by one
        for plot_info in valid_plots_info:
            print(f"\n--- Generating single plot for: {plot_info['hld_name']} ---")
            plot_hld_in_va_space(
                hld_name=plot_info['hld_name'],
                file_hld_path=FILE_HLD_PATH,
                file_va_path=FILE_VA_PATH,
                ss_corpus_path=SS_CORPUS_PATH,
                descriptor_pair=plot_info['pair']
            )
