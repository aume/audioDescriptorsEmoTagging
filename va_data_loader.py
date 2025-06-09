# va_data_loader.py
import pandas as pd
import os

def load_vad_lexicon(filepath):
    """
    Loads a Valence or Arousal lexicon file into a dictionary.
    Assumes tab-separated with word in col 0 and value in col 1.
    Skips the first row (header).
    """
    if not os.path.exists(filepath):
        print(f"Error: Lexicon file not found at '{filepath}'")
        return None
    try:
        df = pd.read_csv(filepath, sep='\t', header=None, names=['word', 'value'], encoding='utf-8')
        df = df.iloc[1:].copy() # Skip the first row which is often a descriptive header
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value'], inplace=True)
        lexicon_dict = df.set_index('word')['value'].to_dict()
        print(f"Loaded {len(lexicon_dict)} entries from '{filepath}'")
        return lexicon_dict
    except Exception as e:
        print(f"An error occurred loading lexicon from '{filepath}': {e}")
        return None

def load_descriptor_pairs(filepath, arousal_lexicon, valence_lexicon):
    """
    Loads descriptor pairs from a file and associates them with VA coordinates.
    Assumes pairs are slash-separated (e.g., "word1/word2").
    """
    descriptor_pairs_list = []
    if not os.path.exists(filepath):
        print(f"Error: Descriptor pairs file not found at '{filepath}'")
        return []
    print(f"Loading descriptor pairs from '{filepath}'...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            words = line.split('/')
            if len(words) < 2:
                continue # Skip malformed lines
            d1_word, d2_word = words[0].strip(), words[1].strip()
            if d1_word not in arousal_lexicon or d1_word not in valence_lexicon or \
               d2_word not in arousal_lexicon or d2_word not in valence_lexicon:
                continue # Skip if words not in lexicon
            
            d1_coord = [valence_lexicon[d1_word], arousal_lexicon[d1_word]]
            d2_coord = [valence_lexicon[d2_word], arousal_lexicon[d2_word]]
            descriptor_pairs_list.append({
                'd1': d1_word, 'd2': d2_word,
                'd1_coord': d1_coord, 'd2_coord': d2_coord
            })
    print(f"Successfully loaded {len(descriptor_pairs_list)} valid descriptor pairs.")
    return descriptor_pairs_list

def load_input_va_points(valence_filepath, arousal_filepath):
    """
    Loads Valence and Arousal values from two separate CSV files
    and combines them into a DataFrame of VA points.
    Assumes each file has a 'value' column and optionally a 'file' column.
    The 'file' column will be taken from the first DataFrame where it's found.
    """
    if not os.path.exists(valence_filepath) or not os.path.exists(arousal_filepath):
        print(f"Error: Input VA files not found. Check paths: '{valence_filepath}', '{arousal_filepath}'")
        return None
    try:
        df_valence = pd.read_csv(valence_filepath, encoding='utf-8')
        df_arousal = pd.read_csv(arousal_filepath, encoding='utf-8')

        if 'value' not in df_valence.columns or 'value' not in df_arousal.columns:
            print("Error: Input CSVs must contain a 'value' column.")
            return None
        
        df_valence['value'] = pd.to_numeric(df_valence['value'], errors='coerce')
        df_arousal['value'] = pd.to_numeric(df_arousal['value'], errors='coerce')
        df_valence.dropna(subset=['value'], inplace=True)
        df_arousal.dropna(subset=['value'], inplace=True)

        min_len = min(len(df_valence), len(df_arousal))
        if min_len == 0:
            print("Error: No valid numeric data found in input VA files after processing.")
            return None
            
        input_va_points_df = pd.DataFrame({
            'valence': df_valence['value'].iloc[:min_len],
            'arousal': df_arousal['value'].iloc[:min_len]
        })

        file_column_name = None
        if 'file' in df_valence.columns:
            input_va_points_df['file'] = df_valence['file'].iloc[:min_len]
            file_column_name = 'file'
        elif 'file' in df_arousal.columns:
            input_va_points_df['file'] = df_arousal['file'].iloc[:min_len]
            file_column_name = 'file'
        
        print(f"Loaded {len(input_va_points_df)} input VA points from CSVs. "
              f"{'File names included.' if file_column_name else 'File names NOT included.'}")
        return input_va_points_df
    except Exception as e:
        print(f"An error occurred loading input VA points: {e}")
        return None