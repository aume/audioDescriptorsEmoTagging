import os
import random
import wave
from mutagen.wave import WAVE
import simpleaudio as sa

# --- Installation Notes ---
# This script requires the 'mutagen' and 'simpleaudio' libraries.
# Install them using pip:
# pip install mutagen
# pip install simpleaudio

def build_file_mapping(corpus_path):
    """
    Scans a directory of WAV files to build a mapping between the file 
    number (from the filename) and the original file name (from metadata).

    Args:
        corpus_path (str): The path to the directory containing the .wav files.

    Returns:
        dict: A dictionary mapping file numbers (as integers) to original 
              file names (as strings). Returns an empty dictionary if the
              path is invalid or no WAV files are found.
    """
    file_mapping = {}
    if not os.path.isdir(corpus_path):
        print(f"Error: The provided path '{corpus_path}' is not a valid directory.")
        return file_mapping

    print(f"Scanning directory: {corpus_path}")
    for filename in os.listdir(corpus_path):
        if filename.endswith(".wav"):
            try:
                file_number = int(os.path.splitext(filename)[0])
                file_path = os.path.join(corpus_path, filename)
                audio = WAVE(file_path)
                if 'TIT2' in audio:
                    original_name = audio['TIT2'].text[0]
                    file_mapping[file_number] = original_name
                else:
                    print(f"Warning: No 'Title' metadata found for {filename}")
            except (ValueError, IndexError) as e:
                print(f"Could not process file {filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with file {filename}: {e}")
                
    return file_mapping

def find_file_in_subdirs(root_path, filename):
    """
    Recursively searches for a file in a directory and its subdirectories.

    Args:
        root_path (str): The base directory to start the search from.
        filename (str): The name of the file to find.

    Returns:
        str: The full path to the file if found, otherwise None.
    """
    for dirpath, _, filenames in os.walk(root_path):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

def test_mapping_and_play_samples(file_mapping, corpus_path, originals_path, num_samples=3):
    """
    Selects random samples, prints the association, and plays both the 
    numbered file and the corresponding original file for verification.

    Args:
        file_mapping (dict): The dictionary mapping file numbers to names.
        corpus_path (str): The path to the directory with the numbered .wav files.
        originals_path (str): The root path for the original named .wav files.
        num_samples (int): The number of random samples to test.
    """
    if not file_mapping:
        print("Cannot run test: The file mapping is empty.")
        return

    sample_keys = list(file_mapping.keys())
    if len(sample_keys) > num_samples:
        sample_keys = random.sample(sample_keys, num_samples)
    else:
        print(f"Warning: The number of samples ({num_samples}) is greater than the "
              f"number of mapped files ({len(file_mapping)}). Testing all files.")

    print("\n--- Starting Verification ---")
    print("Listen to both audio clips for each pair to confirm they are identical.")

    for file_number in sample_keys:
        original_name = file_mapping[file_number]
        
        # --- Play the first file (by number) ---
        numbered_filename = f"{file_number}.wav"
        numbered_filepath = os.path.join(corpus_path, numbered_filename)
        
        print(f"\n--- Testing Pair ---")
        print(f"File Number: {file_number}  <-->  Original Name: '{original_name}'")
        print(f"  (1) Playing numbered file: {numbered_filename}")

        try:
            if os.path.exists(numbered_filepath):
                wave_obj = sa.WaveObject.from_wave_file(numbered_filepath)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            else:
                print(f"      -> Error: Could not find file at path: {numbered_filepath}")
        except Exception as e:
            print(f"      -> Error playing audio file {numbered_filename}: {e}")

        # --- Play the second file (by original name) ---
        original_filename = f"{original_name}.wav"
        print(f"  (2) Now playing original file: {original_filename}")
        
        original_filepath = find_file_in_subdirs(originals_path, original_filename)

        if original_filepath:
            try:
                wave_obj = sa.WaveObject.from_wave_file(original_filepath)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            except Exception as e:
                print(f"      -> Error playing audio file {original_filename}: {e}")
        else:
            print(f"      -> Error: Could not find '{original_filename}' in path: {originals_path}")
            
    print("\n--- Verification Complete ---")


if __name__ == '__main__':
    # --- IMPORTANT ---
    # PLEASE REPLACE THESE WITH THE ACTUAL PATHS TO YOUR AUDIO FILES
    # Path to the numbered files (e.g., "1.wav", "2.wav")
    SS_CORPUS_PATH = "./datasets/EmoSS_Ratings/ES_Corpus"
    
    # Path to the folder containing subfolders of the original named files
    SS_EMO_AUDIO_PATH = "./datasets/Emo-Soundscapes/Emo-Soundscapes-Audio/600_Sounds"

    # --- Step 1: Build the mapping ---
    number_to_name_map = build_file_mapping(SS_CORPUS_PATH)

    if number_to_name_map:
        print("\n--- Generated File Mapping ---")
        for i, (number, name) in enumerate(number_to_name_map.items()):
            if i >= 10:
                print(f"... and {len(number_to_name_map) - 10} more items.")
                break
            print(f"File Number: {number} -> Original Name: {name}")
    else:
        print("\nNo mapping was generated. Please check paths and file integrity.")

    # --- Step 2: Test the mapping ---
    # This will now play both files in each pair for you to compare.
    test_mapping_and_play_samples(
        number_to_name_map, 
        SS_CORPUS_PATH, 
        SS_EMO_AUDIO_PATH, 
        num_samples=3
    )

