
import numpy as np
import pandas as pd
import src

from src.algorithms.incremental.contexts_builder.ContextsBuilder import ContextsBuilder


class ItemsMeanContextsBuilder(ContextsBuilder):

    def __init__(self, users_embeddings: np.ndarray, items_embeddings: np.ndarray, hyperparameters: dict={'num_items_for_mean': 0}):
        super().__init__(users_embeddings, items_embeddings, hyperparameters)

        n_users = self.users_embeddings.shape[0]
        embedding_dim = self.items_embeddings.shape[1]

        self.acum_embs = np.zeros((n_users, embedding_dim))
        self.counts = np.zeros(n_users)


    def generate_new_contexts(self, interactions_df: pd.DataFrame) -> 'list[list[float]]':
        '''
        Gera uma lista de contextos para cada interação, onde o contexto é a média das embeddings dos <num_items_for_mean> itens que o usuário interagiu até o momento.

        params:
            interactions_df: DataFrame contendo interações usuário-item.
            users_embeddings: Embeddings dos usuários.
            items_embeddings: Embeddings dos itens.
            hyperparameters: Hiperparâmetros da geração de contexto. Neste caso, deve ser um dicionário com os seguintes campos:
                - 'num_items_for_mean': Número de itens que serão utilizados para calcular a média das embeddings. A média será calculada apenas com os últimos itens interagidos pelo usuário. Utilize 0 ou negativo para considerar todos os itens interagidos pelo usuário.
        
        return:
            contexts: Lista de contextos para cada interação. Cada contexto é uma lista de floats.
        '''
        contexts = []

        if self.hyperparameters['num_items_for_mean'] <= 0:            
            user_ids = interactions_df[src.COLUMN_USER_ID].values
            item_ids = interactions_df[src.COLUMN_ITEM_ID].values
            
            for user_id, item_id in zip(user_ids, item_ids):
                contexts.append(self.acum_embs[user_id] / max(1, self.counts[user_id]))

                self.acum_embs[user_id] += self.items_embeddings[item_id]
                self.counts[user_id] += 1
        else:
            raise NotImplementedError('item_mean context with num_items_for_mean > 0 is not implemented yet !')

        return contexts