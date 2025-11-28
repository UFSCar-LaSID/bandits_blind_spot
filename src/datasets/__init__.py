import src

import pandas as pd

# -----------------------------------------------------
# Functions to handle missing ratings

def missing_ratings_drop(df_interactions):
    '''
    Remove todas as interações com avaliações faltantes.
    '''
    return df_interactions.dropna(subset=[src.COLUMN_RATING])

def missing_ratings_keep(df_interactions):
    '''
    Mantém todas as interações, mesmo as com avaliações faltantes.
    '''
    return df_interactions


# -----------------------------------------------------
# Functions to define which interactions are positive

def set_positive_middle_rating(df_interactions):
    '''
    Define como positivo todas as interações com avaliação acima do valor médio.
    '''
    max_rating, min_rating = df_interactions[src.COLUMN_RATING].max(), df_interactions[src.COLUMN_RATING].min()
    middle_rating = (max_rating + min_rating) / 2
    is_positive = df_interactions[src.COLUMN_RATING] >= middle_rating
    return is_positive

def set_positive_all(df_interactions):
    '''
    Define como positivo todas as interações.
    '''
    return pd.Series([True] * len(df_interactions), index=df_interactions.index)


from src.datasets.preprocessors.amazon_beauty import preprocess_amazon_beauty
from src.datasets.preprocessors.amazon_books import preprocess_amazon_books
from src.datasets.preprocessors.amazon_games import preprocess_amazon_games
from src.datasets.preprocessors.bestbuy import preprocess_bestbuy
from src.datasets.preprocessors.delicious2k import preprocess_delicious2k
from src.datasets.preprocessors.delicious2k_urlPrincipal import preprocess_delicious2k_urlPrincipal
from src.datasets.preprocessors.ml100k import preprocess_ml100k
from src.datasets.preprocessors.ml25m import preprocess_ml25m
from src.datasets.preprocessors.retailrocket import preprocess_retailrocket

DATASETS_TABLE = pd.DataFrame(
    [[1,     'amazon-beauty',               'E',    missing_ratings_keep,  set_positive_middle_rating, preprocess_amazon_beauty],
     [2,     'amazon-books',                'E',    missing_ratings_keep,  set_positive_middle_rating, preprocess_amazon_books],
     [3,     'amazon-games',                'E',    missing_ratings_keep,  set_positive_middle_rating, preprocess_amazon_games],
     [4,     'bestbuy',                     'I',    missing_ratings_keep,            set_positive_all, preprocess_bestbuy],
     [5,     'delicious2k',                 'I',    missing_ratings_keep,            set_positive_all, preprocess_delicious2k],
     [6,     'delicious2k-urlPrincipal',    'I',    missing_ratings_keep,            set_positive_all, preprocess_delicious2k_urlPrincipal],
     [7,     'ml-100k',                     'E',    missing_ratings_keep,  set_positive_middle_rating, preprocess_ml100k],
     [8,     'ml-25m',                      'E',    missing_ratings_keep,  set_positive_middle_rating, preprocess_ml25m],
     [9,     'retailrocket',                'I',    missing_ratings_keep,            set_positive_all, preprocess_retailrocket]],
    columns=[src.DATASET_ID, src.DATASET_NAME, src.DATASET_TYPE, src.DATASET_MISSING_RATINGS_FUNC, src.DATASET_IS_POSITIVE_FUNC, src.DATASET_PREPROCESS_FUNCTION]
).set_index(src.DATASET_ID)


