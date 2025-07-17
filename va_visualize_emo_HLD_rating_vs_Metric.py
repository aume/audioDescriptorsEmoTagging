import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import glob
from scipy.optimize import minimize_scalar
import sounddevice as sd #pip install sounddevice
import soundfile as sf #pip install soundfile
from emo_file_mapping import build_file_mapping
from va_data_loader import load_vad_lexicon, load_descriptor_pairs

# --- PATHS ---
FILE_HLD_PATH = "./datasets/EmoSS_Ratings"
FILE_AUDIO_PATH = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Audio"

FILE_VA_PATH = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Ratings"
SS_CORPUS_PATH = "./datasets/EmoSS_Ratings/ES_Corpus"
VAD_LEXICON_DIR = "./datasets/NRC-VAD-Lexicon-v2.1/OneFilePerDimension"
AROUSAL_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "arousal-NRC-VAD-Lexicon-v2.1.txt")
VALENCE_LEXICON_PATH = os.path.join(VAD_LEXICON_DIR, "valence-NRC-VAD-Lexicon-v2.1.txt")
DESCRIPTOR_PAIRS_PATH = "./datasets/descriptorPairs.txt"

# Relative base path to audio directory
# we don't know if mixed or not + we don't know the category
AUDIO_BASE_PATH = os.path.join(FILE_AUDIO_PATH, '*', '*')

# Playback control
current_filename = [""]
audio_enabled = [True]

def play_audio(filename):
    if not audio_enabled[0]:
        return
    if (current_filename[0] == filename):
        return
    sd.stop()
    current_filename[0] =filename
    matches = glob.glob(os.path.join(AUDIO_BASE_PATH, filename + ".wav"))
    if not matches:
        print(f"Audio not found: {filename}.wav")
        return

    path = matches[0]
    print("Playing:", path)

    try:
        data, samplerate = sf.read(path, dtype='float32')
        sd.play(data, samplerate)
    except Exception as e:
        print(f"Playback error for {path}: {e}")

def plot_error_against_score_map_opt_tk(root, hld_name, file_hld_path, file_va_path, descriptor_pair, name_to_number_map ):
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
    arousal_df.dropna(subset=['file_number'], inplace=True)
    valence_df.dropna(subset=['file_number'], inplace=True)
    arousal_df['file_number'] = arousal_df['file_number'].astype(int)
    valence_df['file_number'] = valence_df['file_number'].astype(int)
    hld_df['file_number'] = hld_df['file_number'].astype(int)

    va_df = pd.merge(arousal_df[['file_number', 'arousal', 'audio file name']], valence_df[['file_number', 'valence']], on='file_number')
    merged_df = pd.merge(va_df, hld_df, on='file_number')
    if merged_df.empty:
        print(f"No data to plot for {hld_name}.")
        return

    values_hld = merged_df[hld_name].values
    
    # Get original descriptors and their coordinates from the pair
    lexicon_d1, lexicon_d2 = descriptor_pair['d1'], descriptor_pair['d2']
    d1_coord, d2_coord = np.array(descriptor_pair['d1_coord']), np.array(descriptor_pair['d2_coord'])

    resolution = 200
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.stack([xx, yy], axis=-1)

    # FIX: Determine the primary and secondary descriptor for consistent titling.
    # The primary descriptor is the one whose data is being plotted (hld_name).
    if hld_name.lower() == lexicon_d1:
        dist1 = np.linalg.norm(grid_points - d1_coord, axis=-1)
        dist2 = np.linalg.norm(grid_points - d2_coord, axis=-1)
    else:
        dist2 = np.linalg.norm(grid_points - d1_coord, axis=-1)
        dist1 = np.linalg.norm(grid_points - d2_coord, axis=-1)

    xs = ((merged_df['valence'].values + 1) / 2 * (resolution - 1)).astype(int)
    ys = ((merged_df['arousal'].values + 1) / 2 * (resolution - 1)).astype(int)
    xs = np.clip(xs, 0, resolution - 1)
    ys = np.clip(ys, 0, resolution - 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        merged_df['valence'], merged_df['arousal'], c=values_hld,
        cmap='coolwarm', alpha=0.8, zorder=3
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'{hld_name.capitalize()} Score')
    ax.set_title(f'{hld_name}- Rating vs Metric')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.grid(True)

    ax.scatter(d1_coord[0], d1_coord[1], color='green', marker='*', s=200, zorder=5)
    ax.scatter(d2_coord[0], d2_coord[1], color='green', marker='*', s=200, zorder=5)
    ax.annotate(lexicon_d1, (d1_coord[0], d1_coord[1]), textcoords="offset points", xytext=(0,15), ha='center', weight='bold', color ="green")
    ax.annotate(lexicon_d2, (d2_coord[0], d2_coord[1]), textcoords="offset points", xytext=(0,15), ha='center', weight='bold', color ="green")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    showing_error = [False]

    def compute_m1(epsilon):
        s1 = 1 / (epsilon + dist1)
        s2 = 1 / (epsilon + dist2)
        return s1 / (s1 + s2)

    # Create a side panel to include histogram (optional positioning tweak)
    fig_hist, ax_hist = plt.subplots(figsize=(3, 2))
    hist_canvas = FigureCanvasTkAgg(fig_hist, master=root)
    hist_canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.Y, padx=5)

    def update_histogram(error_values):
        ax_hist.clear()
        ax_hist.hist(error_values, bins=20, color='gray', edgecolor='black')
        ax_hist.set_title('Error Histogram')
        ax_hist.set_xlabel('Error')
        ax_hist.set_ylabel('Count')
        fig_hist.tight_layout()
        total_error = np.sum(error_values)
        total_error_label.config(text=f"Total Error: {total_error:.4f}")
        hist_canvas.draw()

    def update_plot(log_eps):
        epsilon = 10 ** float(log_eps)
        M1 = compute_m1(epsilon)
        pred = M1[ys, xs]
        values = np.abs(pred - values_hld) if showing_error[0] else values_hld

        scatter.set_array(values)
        scatter.set_cmap('magma' if showing_error[0] else 'coolwarm')
        cbar.set_label('|Rating - Metric| Error' if showing_error[0] else f'{hld_name.capitalize()} Score')

        if hasattr(update_plot, 'im_artist') and update_plot.im_artist in ax.images:
            update_plot.im_artist.remove()
        update_plot.im_artist = ax.imshow(
            M1, extent=[-1, 1, -1, 1], origin='lower', cmap='coolwarm', vmin=0, vmax=1, alpha=1.0, zorder=0
        )
        if showing_error[0]:
            update_histogram(values)
        else:
            update_histogram([])
        canvas.draw_idle()

    def toggle():
        showing_error[0] = not showing_error[0]
        toggle_button.config(text="Show HLD Rating" if showing_error[0] else "Show Error")
        update_plot(epsilon_slider.get())

    def optimize():
        def total_error(log_eps):
            epsilon = 10 ** log_eps
            M1 = compute_m1(epsilon)
            pred = M1[ys, xs]
            return np.abs(pred - values_hld).sum()
        result = minimize_scalar(total_error, bounds=(-4, 0), method='bounded')
        if result.success:
            epsilon_slider.set(result.x)

    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # Label to display filename on hover
    filename_var = tk.StringVar()
    filename_label = tk.Label(control_frame, textvariable=filename_var, font=('Arial', 10, 'italic'))
    filename_label.pack(side=tk.TOP, padx=10)
    
    audio_var = tk.IntVar(value=1)
    checkbox = tk.Checkbutton(control_frame, text="Enable Audio", variable=audio_var,
                               command=lambda: audio_enabled.__setitem__(0, bool(audio_var.get())))
    checkbox.pack(side=tk.LEFT, padx=10)

    total_error_label = tk.Label(control_frame, font=('Arial', 10, 'italic'))
    total_error_label.pack(side=tk.TOP, padx=10)

    tk.Label(control_frame, text="log10(Epsilon)").pack(side=tk.LEFT, padx=5)
    epsilon_slider = tk.Scale(control_frame, from_=-4, to=0, resolution=0.01, orient=tk.HORIZONTAL, length=300,
                              command=update_plot)
    epsilon_slider.set(0.0)
    epsilon_slider.pack(side=tk.LEFT)

    toggle_button = tk.Button(control_frame, text="Show Error", command=toggle)
    toggle_button.pack(side=tk.LEFT, padx=10)

    opt_button = tk.Button(control_frame, text="Optimize Epsilon", command=optimize)
    opt_button.pack(side=tk.LEFT, padx=10)

    # Store dot-to-file mapping
    file_paths = merged_df['audio file name'].tolist()

    # Use event to get closest point
    hovered_index = [None]

    # Enable hover
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", 
                        bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    sc_coords = scatter.get_offsets()

    # Called when cursor moves
    def on_hover(event):
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                index = ind["ind"][0]
                hovered_index[0] = index
                pos = sc_coords[index]
                annot.xy = pos
                filename = file_paths[index]
                annot.set_text(filename)
                annot.set_visible(True)
                filename_var.set(filename)
                play_audio(filename)
                canvas.draw_idle()
            else:
                annot.set_visible(False)
                canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", on_hover)

    update_plot(0.0)

if __name__ == '__main__':
    
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

    number_to_name_map = build_file_mapping(SS_CORPUS_PATH)
    name_to_number_map = {v: k for k, v in number_to_name_map.items()}

    root = tk.Tk()
    root.title("HLD vs VA Plot Viewer")
    current_index = [0]  # mutable index so nested functions can modify it

    def add_navigation_buttons():
        nav_frame = tk.Frame(root)
        nav_frame.pack(side=tk.BOTTOM, pady=5)

        prev_button = tk.Button(nav_frame, text="<< Prev", command=go_prev)
        prev_button.pack(side=tk.LEFT, padx=5)
        next_button = tk.Button(nav_frame, text="Next >>", command=go_next)
        next_button.pack(side=tk.LEFT, padx=5)

    def show_plot(index):
        for widget in root.winfo_children():
            widget.destroy()
        plot_info = valid_plots_info[index]
        plot_error_against_score_map_opt_tk(
            root=root,
            hld_name=plot_info['hld_name'],
            file_hld_path=FILE_HLD_PATH,
            file_va_path=FILE_VA_PATH,
            descriptor_pair=plot_info['pair'],
            name_to_number_map = name_to_number_map
        )
        add_navigation_buttons()

    def go_prev():
        if current_index[0] > 0:
            current_index[0] -= 1
            show_plot(current_index[0])

    def go_next():
        if current_index[0] < len(valid_plots_info) - 1:
            current_index[0] += 1
            show_plot(current_index[0])

    show_plot(current_index[0])
    root.mainloop()

