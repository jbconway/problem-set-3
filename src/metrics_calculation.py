'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Calculate micro-averaged precision:  TP / (TP + FP)
    total_tp = sum(genre_tp_counts.get(g, 0) for g in genre_list) #total true positives across all genres
    total_fp = sum(genre_fp_counts.get(g, 0) for g in genre_list) #  total false positives across all genres
    total_fn = sum(genre_true_counts.get(g, 0) - genre_tp_counts.get(g, 0) for g in genre_list) # total false negatives across all genres

    # Calculate micro-averaged precision: TP / (TP + FP)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    # Calculate micro-averaged recall: TP / (TP + FN)
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    # Calculate micro-averaged F1 score
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0

    # Macro lists
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for genre in genre_list:
        tp = genre_tp_counts.get(genre, 0) # true positives for this genre
        fp = genre_fp_counts.get(genre, 0) # false positives
        fn = genre_true_counts.get(genre, 0) - tp # false negatives
        # Calculate precision, recall, and F1 for this genre
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # Append per-genre metrics to lists
        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)

    # Flatten the return values so main.py unpacking works without changes
    return micro_precision, micro_recall, micro_f1, macro_precisions, macro_recalls, macro_f1s

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    pred_rows = []
    true_rows = []

    import ast

    for _, row in model_pred_df.iterrows():
        pred_genres = {row['predicted']}
        true_genres = set(ast.literal_eval(row['actual genres']))
    # For each genre in genre_list, assign 1 if predicted/true, else 0
        pred_row = [1 if genre in pred_genres else 0 for genre in genre_list]
        true_row = [1 if genre in true_genres else 0 for genre in genre_list]

        pred_rows.append(pred_row)
        true_rows.append(true_row)
     # Convert lists of indicator vectors into DataFrames 
    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)
    true_matrix = pd.DataFrame(true_rows, columns=genre_list)
    # Use sklearn to calculate macro-averaged precision, recall, and F1 score
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='macro', zero_division=0)
    # Use sklearn to calculate micro-averaged precision, recall, and F1 score
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='micro', zero_division=0)

    # Return all six values flat, so main.py unpacking works as is
    return macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1
