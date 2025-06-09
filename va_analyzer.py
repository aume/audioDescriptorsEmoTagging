# va_analyzer.py
import pandas as pd
from va_utils import get_localized_pair_scores

def perform_descriptor_analysis(input_va_points_df, descriptor_pairs):
    """
    Calculates descriptor values for each input VA point based on descriptor pairs.
    
    Args:
        input_va_points_df (pd.DataFrame): DataFrame with 'valence', 'arousal', and optionally 'file' columns.
        descriptor_pairs (list): List of dictionaries, each containing 'd1', 'd2', 'd1_coord', 'd2_coord'.
        
    Returns:
        pd.DataFrame: A DataFrame containing input VA points, file names, and calculated descriptor scores.
                      Returns an empty DataFrame if no descriptor pairs are provided.
    """
    results_list = []
    if not descriptor_pairs:
        print("No descriptor pairs provided for analysis.")
        return pd.DataFrame()

    print("\nCalculating descriptor values for each input VA point...")
    for idx, row in input_va_points_df.iterrows():
        target_va_point = [row['valence'], row['arousal']]
        
        row_results = {
            'input_point_idx': idx,
            'input_valence': target_va_point[0],
            'input_arousal': target_va_point[1]
        }
        
        if 'file' in row:
            row_results['file'] = row['file']

        for pair_info in descriptor_pairs:
            d1_word = pair_info['d1']
            d2_word = pair_info['d2']
            d1_coord = pair_info['d1_coord']
            d2_coord = pair_info['d2_coord']

            score_d1, score_d2 = get_localized_pair_scores(
                target_va_point, d1_coord, d2_coord
            )
            
            row_results[f'score_{d1_word}'] = score_d1
            row_results[f'score_{d2_word}'] = score_d2

        results_list.append(row_results)
    
    print("Analysis complete.")
    return pd.DataFrame(results_list)