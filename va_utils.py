# va_utils.py
import numpy as np

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two 2D points."""
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_localized_pair_scores(target_va_coord, concept1_coord, concept2_coord):
    """
    Calculates how 'concept1-like' and 'concept2-like' a target VA coordinate is,
    with scores between 0 and 1, locally scaled based on the distances to the two concepts.
    
    Args:
        target_va_coord (list/tuple): The [valence, arousal] coordinate to analyze.
        concept1_coord (list/tuple): The [valence, arousal] coordinate of the first concept word.
        concept2_coord (list/tuple): The [valence, arousal] coordinate of the second concept word.
        
    Returns:
        tuple: (concept1_score, concept2_score). Each score is between 0 and 1.
    """
    raw_dist1 = euclidean_distance(target_va_coord, concept1_coord)
    raw_dist2 = euclidean_distance(target_va_coord, concept2_coord)

    if raw_dist1 == 0 and raw_dist2 == 0:
        return (0.5, 0.5) 
    elif raw_dist1 == 0:
        return (1.0, 0.0) 
    elif raw_dist2 == 0:
        return (0.0, 1.0)
        
    similarity1 = 1 / (1 + raw_dist1)
    similarity2 = 1 / (1 + raw_dist2)

    total_similarity = similarity1 + similarity2
    score1 = similarity1 / total_similarity
    score2 = similarity2 / total_similarity

    return score1, score2