from abc import ABC
import pandas as pd
import numpy as np
from typing import Union
from mab2rec import LearningPolicy
from scipy.sparse import csr_matrix
import src

class Mab2RecRecommenderOptimized(ABC):
    '''
    Classe base para os recomendadores da biblioteca [mab2rec](https://github.com/fidelity/mab2rec).

    Essa classe é responsável por encapsular a lógica de inicialização e treinamento dos algoritmos da biblioteca mab2rec.

    Essa classe não deve ser utilizada diretamente. Para utilizar um algoritmo da biblioteca mab2rec, utilize uma das classes filhas dessa classe.

    Para implementar um novo algoritmo da biblioteca mab2rec, basta na classe filha implementar o método __init__ chamando o super().__init__ e inicializar o atributo self.recommender com o algoritmo da biblioteca mab2rec (pelo menos, a ideia inicial é ser simples assim).
    '''

    def __init__(self, num_users: int, num_items: int, num_features: int, user_column: str=src.COLUMN_USER_ID, item_column: str=src.COLUMN_ITEM_ID, rating_column: str=src.COLUMN_RATING, hyperparameters: dict={}, logger = None):
        '''
        Inicializa o recomendador.

        params:
            user_column: Nome da coluna que representa o usuário.
            item_column: Nome da coluna que representa o item.
            rating_column: Nome da coluna que representa a avaliação.
        '''
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.num_users = num_users
        self.num_features = num_features
        self.num_items = num_items

        self.interactions_by_user: pd.DataFrame = None
        self.recommender: LearningPolicy = None
        if src.EXECUTION_MODE == 'gpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.exclude_mask = csr_matrix(([], ([], [])), shape=(self.num_users, self.num_items), dtype=bool).tolil()

    def train(self, interactions_df: pd.DataFrame, contexts):
        '''
        Treina "do zero" o recomendador com base nas interações passadas. Utilizar apenas na primeira chamada de treinamento, caso deseje treinar incrementalmente.

        params:
            interactions_df: DataFrame contendo as interações usuário-item.
        '''
        self.exclude_mask[interactions_df[self.user_column], interactions_df[self.item_column]] = True
        self.recommender.fit(
            decisions=interactions_df[self.item_column],
            rewards=interactions_df[self.rating_column],
            contexts=contexts
        )

    def partial_train(self, interactions_df: pd.DataFrame, contexts):
        '''
        Treina o recomendador incrementalmente com base nas interações passadas. Deve ser utilizado após a primeira chamada de treinamento, o novo conhecimento será incorporado ao modelo, sem esquecer o conhecimento anterior.

        params:
            interactions_df: DataFrame contendo as interações usuário-item.
        '''
        self.exclude_mask[interactions_df[self.user_column], interactions_df[self.item_column]] = True
        self.recommender.partial_fit(
            decisions=interactions_df[self.item_column],
            rewards=interactions_df[self.rating_column],
            contexts=contexts
        )
    
    def recommend(self, users_ids: 'Union[list[int], np.ndarray]', contexts) -> 'tuple[list[int], list[float]]':
        '''
        Gera recomendações para uma lista de usuários.

        params:
            users_ids: Lista de IDs dos usuários para os quais deseja-se gerar recomendações.
            topn: Número máximo de recomendações a serem geradas por `user_id`.

        returns:
            Tupla contendo dois arrays: o primeiro contém os IDs dos itens recomendados e o segundo contém a pontuação de cada item.
        '''
        return self.recommender.recommend(contexts, self.exclude_mask[users_ids].nonzero(), apply_sigmoid=False, return_scores=True)