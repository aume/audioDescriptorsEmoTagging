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
    ax.set_yticks(np.arange(-1.0, 1.0, 0.2))
    ax.set_xticks(np.arange(-1.0, 1.0, 0.2))

    # Set explicit limits for the Valence-Arousal space (NRC-VAD typical)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0) # Arousal is usually [-1, 1] for NRC-VAD
    
    ax.grid(True, linestyle='--', alpha=0.7)
    #plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH+'desc_emo_va_space_visualization.png')
    plt.show()


def plot_va_space_radial(input_va_points_df, descriptor_pairs, title="Input VA"):
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

    # --- Helper function for coordinate conversion (Valence as X, Arousal as Y) ---
    def cartesian_to_polar(valence, arousal):
        # Radius is the Euclidean distance from the origin (magnitude of the emotional state)
        r = np.hypot(valence, arousal) 
        # Angle is calculated using arctan2(y, x) -> arctan2(arousal, valence)
        theta = np.arctan2(arousal, valence)
        return theta, r

    # --- Plot VA values from input_va_points_df ---
    # input_thetas, input_rs = cartesian_to_polar(input_va_points_df['valence'], input_va_points_df['arousal'])
    # ax.plot(input_thetas, input_rs,
    #         marker='o', linestyle='none', color='blue', markersize=8, alpha=0.7,
    #         label='Input VA Points')

    # --- Plot descriptor pairs ---
    for item in descriptor_pairs:
        d1_theta, d1_r = cartesian_to_polar(item['d1_coord'][0], item['d1_coord'][1])
        d2_theta, d2_r = cartesian_to_polar(item['d2_coord'][0], item['d2_coord'][1])
        
        # Plot the line connecting descriptor pair coordinates
        ax.plot([d1_theta, d2_theta], [d1_r, d2_r],
                marker='o', linestyle='-', markersize=8, linewidth=2, alpha=0.7)
        
        # Annotate d1
        # Adjust xytext based on angle and radius for better readability
        ax.annotate(item['d1'], xy=(d1_theta, d1_r),
                    xytext=(d1_theta, d1_r + 0.08), textcoords='data', ha='center', va='bottom',
                    fontsize=9, color='darkred')
        # Annotate d2
        ax.annotate(item['d2'], xy=(d2_theta, d2_r),
                    xytext=(d2_theta, d2_r - 0.08), textcoords='data', ha='center', va='top',
                    fontsize=9, color='darkgreen')

    # --- Configure radial plot ---
    # Max possible radius is sqrt(1^2 + 1^2) = sqrt(2) approx 1.414
    max_radius = np.sqrt(1**2 + 1**2)
    ax.set_ylim(0, max_radius * 1.0) # Add a small buffer
    #ax.set_yticks(np.arange(0, max_radius + 0.2, 0.2)) # Ticks for magnitude
    ax.set_yticklabels([f'{y:.1f}' for y in np.arange(0, max_radius + 0.2, 0.2)]) # Label magnitude

    # Set angular labels (representing directions in VA space)
    # Label the four cardinal directions for intuition
    # Note: arctan2 output is in radians from -pi to pi
    ax.set_xticks(np.array([0, np.pi/2, np.pi, 3*np.pi/2]))
    ax.set_xticklabels(['Valence (+)', 'Arousal (+)', 'Valence (-)', 'Arousal (-)'])
    
    # Adjust the theta_offset if you want a specific axis aligned (e.g., Valence (+) at the top)
    # ax.set_theta_offset(np.pi / 2) # Example: Rotates Valence (+) to the top

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(title, va='bottom')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH + 'desc_emo_radial_bipolar_va_space_visualization.png')
    plt.show()