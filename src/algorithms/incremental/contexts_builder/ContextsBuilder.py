
import pandas as pd
import numpy as np

class ContextsBuilder:

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict):
        self.users_embeddings = users_embeddings
        self.items_embeddings = items_embeddings
        self.hyperparameters = hyperparameters

    def generate_new_contexts(self, interactions_df: pd.DataFrame) -> 'list[list[float]]':
        pass
