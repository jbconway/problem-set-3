'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Load the model predictions and genres data
    model_pred_df = pd.read_csv("data/prediction_model_03.csv")
    genres_df = pd.read_csv("data/genres.csv")

    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Extract the list of unique genres from the genres DataFrame
    # then sort it alphabetically 
    genre_list = sorted(genres_df['genre'].unique().tolist())

    # create counts
    genre_true_counts = {genre: 0 for genre in genre_list} # how many times each genre actually appears
    genre_tp_counts = {genre: 0 for genre in genre_list} # how many times each genre was correctly predicted
    genre_fp_counts = {genre: 0 for genre in genre_list} # how many times each genre was predicted but not actually present (false positives)

    for _, row in model_pred_df.iterrows():
        pred_genres = {row['predicted']}  

        # Convert actual genres string into list using ast
        true_genres = set(ast.literal_eval(row['actual genres']))

    # For every true genre in the actual genres increment its count in genre_true_counts
        for genre in true_genres:
            if genre in genre_true_counts:
                genre_true_counts[genre] += 1

    # for every predicted genre if it is in the true genres, increment the true positive count for that genre
    # or, increment the false positive count 
        for genre in pred_genres:
            if genre in true_genres:
                genre_tp_counts[genre] += 1
            else:
                genre_fp_counts[genre] += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts