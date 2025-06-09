# va_plotter.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Used for DataFrame type hint, not directly for plotting data here

FIGURE_PATH = './figures/'

def plot_va_space(input_va_points_df, descriptor_pairs, title="Input VA Points & Descriptor Pairs in VA Space"):
    """
    Plots input VA points and descriptor pairs in the Valence-Arousal space.
    
    Args:
        input_va_points_df (pd.DataFrame): DataFrame with 'valence' and 'arousal' columns.
        descriptor_pairs (list): List of dictionaries, each containing 'd1_coord', 'd2_coord', 'd1', 'd2'.
        title (str): Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot VA values from predictions
    ax.plot(input_va_points_df['valence'], input_va_points_df['arousal'],
            marker='o', linestyle='none', color='blue', markersize=8, alpha=0.7,
            label='Input VA Points')

    # Plot descriptor pairs
    for item in descriptor_pairs:
        X = [item['d1_coord'][0], item['d2_coord'][0]]
        Y = [item['d1_coord'][1], item['d2_coord'][1]]
        label = f"{item['d1']}/{item['d2']}"
        ax.plot(X, Y, marker='o', linestyle='-', markersize=8, linewidth=2, label=label, alpha=0.7)
        
        # Annotate d1
        ax.annotate(item['d1'], xy=(item['d1_coord'][0], item['d1_coord'][1]),
                    xytext=(5, 5), textcoords='offset points', ha='left', va='bottom',
                    fontsize=9, color='darkred')
        # Annotate d2
        ax.annotate(item['d2'], xy=(item['d2_coord'][0], item['d2_coord'][1]),
                    xytext=(-5, -5), textcoords='offset points', ha='right', va='top',
                    fontsize=9, color='darkgreen')

    # Add central crosshair for reference
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, label='Neutral Valence')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, label='Neutral Arousal')

    # Set ticks based on standard VA ranges
    ax.set_yticks(np.arange(-1.0, 1.1, 0.2))
    ax.set_xticks(np.arange(-1.0, 1.1, 0.2))

    # Set explicit limits for the Valence-Arousal space (NRC-VAD typical)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05) # Arousal is usually [0, 1] for NRC-VAD
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH+'desc_emo_va_space_visualization.png')
    plt.show()