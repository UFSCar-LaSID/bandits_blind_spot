
import pandas as pd
import numpy as np

from typing import Union

import src
from src.datasets.Dataset import Dataset
import json
import os


def save_dict_as_json(data_dict: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fp:
        json.dump(data_dict, fp, indent=4)


def save_dataframe(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)


def save_val_recommendations(val_interactions_df: pd.DataFrame, recommendations_ids: 'Union[list, np.ndarray]', recommendations_scores: 'Union[list, np.ndarray]', dataset: Dataset, save_path: str) -> pd.DataFrame:
    if isinstance(recommendations_scores, np.ndarray):
        recommendations_scores = recommendations_scores.tolist()
    
    val_recs_df = pd.DataFrame({
        src.COLUMN_USER_ID: dataset.decode_users_ids(val_interactions_df[src.COLUMN_USER_ID].values),
        src.COLUMN_ITEM_ID: dataset.decode_items_ids(val_interactions_df[src.COLUMN_ITEM_ID].values),
        src.COLUMN_RATING: val_interactions_df[src.COLUMN_RATING].values,
        src.COLUMN_IS_POSITIVE: val_interactions_df[src.COLUMN_IS_POSITIVE].values,
        src.COLUMN_DATETIME: val_interactions_df[src.COLUMN_DATETIME].values,
        src.COLUMN_RECOMMENDATIONS: dataset.decode_items_ids(recommendations_ids),
        src.COLUMN_SCORES: recommendations_scores
    })
    save_dataframe(val_recs_df, save_path)
    return val_recs_df


class TestRecommendationsSaver:

    def __init__(self):
        self.current_window_number = 0

        self.users_ids_full_array = np.array([], dtype=int)
        self.items_ids_full_array = np.array([], dtype=int)
        self.ratings_full_array = np.array([], dtype=float)
        self.is_positive_full_array = np.array([], dtype=bool)
        self.datetime_full_array = np.array([], dtype="datetime64[ns]")
        self.window_numbers_full_array = np.array([], dtype=int)
        self.recommendations_ids_full_array = None
        self.recommendations_scores_full_array = None

    def add_new_test_window(self, test_window_interactions_df: pd.DataFrame, recommendations_ids: 'Union[list, np.ndarray]', recommendations_scores: 'Union[list, np.ndarray]'):
        self.users_ids_full_array = np.concatenate((self.users_ids_full_array, test_window_interactions_df[src.COLUMN_USER_ID].values))
        self.items_ids_full_array = np.concatenate((self.items_ids_full_array, test_window_interactions_df[src.COLUMN_ITEM_ID].values))
        self.ratings_full_array = np.concatenate((self.ratings_full_array, test_window_interactions_df[src.COLUMN_RATING].values))
        self.is_positive_full_array = np.concatenate((self.is_positive_full_array, test_window_interactions_df[src.COLUMN_IS_POSITIVE].values))
        self.datetime_full_array = np.concatenate((self.datetime_full_array, test_window_interactions_df[src.COLUMN_DATETIME].values))

        self.window_numbers_full_array = np.concatenate((self.window_numbers_full_array, np.repeat(self.current_window_number, len(recommendations_ids))))

        if self.current_window_number == 0:
            self.recommendations_ids_full_array = np.array(recommendations_ids)
            self.recommendations_scores_full_array = np.array(recommendations_scores)
        else:
            self.recommendations_ids_full_array = np.concatenate((self.recommendations_ids_full_array, recommendations_ids))
            self.recommendations_scores_full_array = np.concatenate((self.recommendations_scores_full_array, recommendations_scores))

        self.current_window_number += 1
    
    def save(self, save_path: str, dataset: Dataset) -> pd.DataFrame:
        test_recs_df = pd.DataFrame({
            src.COLUMN_USER_ID: dataset.decode_users_ids(self.users_ids_full_array),
            src.COLUMN_ITEM_ID: dataset.decode_items_ids(self.items_ids_full_array),
            src.COLUMN_RATING: self.ratings_full_array,
            src.COLUMN_IS_POSITIVE: self.is_positive_full_array,
            src.COLUMN_DATETIME: self.datetime_full_array,
            src.COLUMN_WINDOW_NUMBER: self.window_numbers_full_array,
            src.COLUMN_RECOMMENDATIONS: dataset.decode_items_ids(self.recommendations_ids_full_array),
            src.COLUMN_SCORES: self.recommendations_scores_full_array.tolist()
        })
        save_dataframe(test_recs_df, save_path)
        return test_recs_df
