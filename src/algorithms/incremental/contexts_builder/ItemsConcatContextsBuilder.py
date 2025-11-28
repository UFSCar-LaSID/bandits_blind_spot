
import numpy as np
import pandas as pd
import src

from src.algorithms.incremental.contexts_builder.ContextsBuilder import ContextsBuilder


class ItemsConcatContextsBuilder(ContextsBuilder):

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict={'window_size': 5}):
        super().__init__(users_embeddings, items_embeddings, hyperparameters)

        self.users_current_info = {}

    def generate_new_contexts(self, interactions_df: pd.DataFrame) -> 'list[list[float]]':
        '''
        Gera uma lista de contextos para cada interação, onde o contexto é a concatenação das embeddings dos últimos <window_size> itens que o usuário interagiu.

        params:
            interactions_df: DataFrame contendo interações usuário-item.
            users_embeddings: Embeddings dos usuários.
            items_embeddings: Embeddings dos itens.
            hyperparameters: Hiperparâmetros da geração de contexto. Neste caso, deve ser um dicionário com os seguintes campos:
                - 'window_size': Tamanho da janela de itens que serão utilizados para gerar o contexto (quantidade de itens que serão concatenados).
        
        return:
            contexts: Lista de contextos para cada interação. Cada contexto é uma lista de floats.
        '''
        contexts = []
        window_size = self.hyperparameters['window_size']

        for _, row in interactions_df.iterrows():
            user_id = row[src.COLUMN_USER_ID]
            item_id = row[src.COLUMN_ITEM_ID]

            if user_id not in self.users_current_info:
                self.users_current_info[user_id] = np.zeros((window_size, self.items_embeddings.shape[1]))
            
            contexts.append(self.users_current_info[user_id].flatten())
            
            self.users_current_info[user_id][1:] = self.users_current_info[user_id][:-1]
            self.users_current_info[user_id][0] = self.items_embeddings[item_id][:self.items_embeddings.shape[1]]

        return contexts