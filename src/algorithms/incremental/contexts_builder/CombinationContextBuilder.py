
import numpy as np
import pandas as pd
from src.algorithms.incremental.contexts_builder.ContextsBuilder import ContextsBuilder
from src.algorithms.incremental.contexts_builder.ItemsMeanContextsBuilder import ItemsMeanContextsBuilder
from src.algorithms.incremental.contexts_builder.UserContextsBuilder import UserContextsBuilder
from src.algorithms.incremental.contexts_builder.ItemsConcatContextsBuilder import ItemsConcatContextsBuilder


class CombinationContextsBuilder(ContextsBuilder):

    def generate_new_contexts(self, interactions_df: pd.DataFrame):
        separated_contexts = []
        for context_builder in self.contexts_builders:
            separated_contexts += context_builder.generate_new_contexts(interactions_df)
        
        contexts = []
        for i in range(len(interactions_df)):
            context = []
            for separated_context in separated_contexts:
                context += separated_context[i]
            contexts.append(context)
        
        return contexts


class ItemsMeanUserContextsBuilder(CombinationContextsBuilder):

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict={'num_items_for_mean': 0}):
        super().__init__(users_embeddings, items_embeddings, hyperparameters)
        self.contexts_builders = [
            ItemsMeanContextsBuilder(users_embeddings, items_embeddings, hyperparameters),
            UserContextsBuilder(users_embeddings, items_embeddings, hyperparameters)
        ]

class ItemsConcatUserContextsBuilder(CombinationContextsBuilder):

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict={'window_size': 5}):
        super().__init__(users_embeddings, items_embeddings, hyperparameters)
        self.contexts_builders = [
            ItemsConcatContextsBuilder(users_embeddings, items_embeddings, hyperparameters),
            UserContextsBuilder(users_embeddings, items_embeddings, hyperparameters)
        ]

class ItemsConcatItemsMeanContextsBuilder(CombinationContextsBuilder):

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict={'num_items_for_mean': 0, 'window_size': 5}):
        super().__init__(users_embeddings, items_embeddings, hyperparameters)
        self.contexts_builders = [
            ItemsMeanContextsBuilder(users_embeddings, items_embeddings, hyperparameters),
            ItemsConcatContextsBuilder(users_embeddings, items_embeddings, hyperparameters)
        ]

class ItemsConcatItemsMeanUserContextsBuilder(CombinationContextsBuilder):

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict={'num_items_for_mean': 0, 'window_size': 5}):
        super().__init__(users_embeddings, items_embeddings, hyperparameters)
        self.contexts_builders = [
            ItemsMeanContextsBuilder(users_embeddings, items_embeddings, hyperparameters),
            UserContextsBuilder(users_embeddings, items_embeddings, hyperparameters),
            ItemsConcatContextsBuilder(users_embeddings, items_embeddings, hyperparameters)
        ]