
import numpy as np
import pandas as pd
import src

from src.algorithms.incremental.contexts_builder.ContextsBuilder import ContextsBuilder


class UserContextsBuilder(ContextsBuilder):

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict={}):
        super().__init__(users_embeddings, items_embeddings, hyperparameters)

    def generate_new_contexts(self, interactions_df: pd.DataFrame) -> 'list[list[float]]':
        '''
        Gera uma lista de contextos para cada interação, onde o contexto é a embedding do usuário.

        params:
            interactions_df: DataFrame contendo interações usuário-item.
            users_embeddings: Embeddings dos usuários.
            items_embeddings: Embeddings dos itens.
            hyperparameters: Hiperparâmetros da geração de contexto. Neste caso, não há nenhum hiperparâmetro (dicionário vazio).
        
        return:
            contexts: Lista de contextos para cada interação. Cada contexto é uma lista de floats.
        '''
        contexts = []

        for _, row in interactions_df.iterrows():
            user_id = row[src.COLUMN_USER_ID]
            contexts.append(self.users_embeddings[user_id][:self.users_embeddings.shape[1]].astype(np.double))

        return contexts