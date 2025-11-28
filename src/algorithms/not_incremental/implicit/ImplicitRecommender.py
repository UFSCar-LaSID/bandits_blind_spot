
from abc import ABC
from implicit.cpu.matrix_factorization_base import MatrixFactorizationBase
import numpy as np
import os
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Union
import src

class ImplicitRecommender(ABC):
    '''
    Classe base para os recomendadores da biblioteca [Implicit](https://benfred.github.io/implicit/index.html).

    Essa classe é responsável por encapsular a lógica de inicialização e treinamento dos algoritmos da biblioteca Implicit.

    Essa classe não deve ser utilizada diretamente. Para utilizar um algoritmo da biblioteca Implicit, utilize uma das classes filhas dessa classe.

    Para implementar um novo algoritmo da biblioteca Implicit, basta na classe filha implementar o método __init__ chamando o super().__init__ e inicializar o atributo self.recommender com o algoritmo da biblioteca Implicit (pelo menos, a ideia inicial é ser simples assim).
    '''

    def __init__(self, user_column: str=src.COLUMN_USER_ID, item_column: str=src.COLUMN_ITEM_ID, rating_column: str=src.COLUMN_RATING, hyperparameters: dict=dict()):
        '''
        Inicializa o recomendador.

        params:
            user_column: Nome da coluna que representa o usuário.
            item_column: Nome da coluna que representa o item.
            rating_column: Nome da coluna que representa a avaliação.
            negative_strategy: Estratégia para tratar amostras negativas. Varia de acordo com a classe filha.
        '''
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        self.user_column = user_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.negative_strategy = hyperparameters['negative_strategy'] if 'negative_strategy' in hyperparameters else None
        self.model_hyperparameters = {k: v for k, v in hyperparameters.items() if k != 'negative_strategy'}
        self.recommender: MatrixFactorizationBase = None


    def _get_ratings(self, interactions_df: pd.DataFrame) -> np.ndarray:
        '''
        Retorna as avaliações das interações.

        params:
            interactions_df: DataFrame contendo as interações usuário-item.

        returns:
            Array contendo as avaliações das interações.
        '''
        return interactions_df[self.rating_column].values
    

    def train(self, interactions_df: pd.DataFrame, num_users: int, num_items: int, verbose: bool=False):
        '''
        Treina o recomendador com base nas interações passadas.

        params:
            interactions_df: DataFrame contendo as interações usuário-item.
            num_users: numero de usuários em toda a base de dados
            num_items: numero de usuários em toda a base de dados
        '''
        ratings = self._get_ratings(interactions_df) if self.rating_column in interactions_df.columns else np.ones(len(interactions_df))
        self.sparse_matrix = csr_matrix((
            ratings, 
            (interactions_df[self.user_column], interactions_df[self.item_column])), 
            shape=(num_users, num_items)
        )
        self.recommender.fit(self.sparse_matrix, show_progress=verbose)


    def recommend(self, users_ids: 'Union[list[int], np.ndarray]', topn=src.TOP_N) -> 'tuple[np.ndarray[int], np.ndarray[float]]':
        '''
        Gera recomendações para uma lista de usuários.

        params:
            users_ids: Lista de IDs dos usuários para os quais deseja-se gerar recomendações.
            topn: Número máximo de recomendações a serem geradas por `user_id`.

        returns:
            Tupla contendo dois arrays: o primeiro contém os IDs dos itens recomendados e o segundo contém a pontuação de cada item.
        '''
        return self.recommender.recommend(
            userid=users_ids, 
            user_items=self.sparse_matrix[users_ids],
            N=topn,
            filter_already_liked_items=True
        )
    
    @property
    def users_embeddings(self) -> np.ndarray:
        '''
        Retorna os embeddings dos usuários.

        returns:
            Array contendo os embeddings dos usuários.
        '''
        if not isinstance(self.recommender.user_factors, np.ndarray):
            return self.recommender.user_factors.to_numpy()
        return self.recommender.user_factors

    @property
    def items_embeddings(self) -> np.ndarray:
        '''
        Retorna os embeddings dos itens.

        returns:
            Array contendo os embeddings dos itens.
        '''
        if not isinstance(self.recommender.item_factors, np.ndarray):
            return self.recommender.item_factors.to_numpy()
        return self.recommender.item_factors
    
    def save_embeddings(self, save_path: str):
        '''
        Salva os embeddings dos usuários e itens em arquivo npy.

        params:
            save_path: Caminho onde os embeddings devem ser salvos.
        '''
        np.save(os.path.join(save_path, src.FILE_USERS_EMBEDDINGS), self.users_embeddings)
        np.save(os.path.join(save_path, src.FILE_ITEMS_EMBEDDINGS), self.items_embeddings)