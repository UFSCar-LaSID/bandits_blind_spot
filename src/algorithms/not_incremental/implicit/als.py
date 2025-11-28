import implicit
import numpy as np
import pandas as pd
import src

from src.algorithms.not_incremental.implicit.ImplicitRecommender import ImplicitRecommender

class AlternatingLeastSquares(ImplicitRecommender):

    def __init__(self, user_column: str=src.COLUMN_USER_ID, item_column: str=src.COLUMN_ITEM_ID, rating_column: str=src.COLUMN_RATING, hyperparameters: dict=dict()):
        '''
        Inicializa o recomendador.

        params:
            user_column: Nome da coluna que representa o usuário.
            item_column: Nome da coluna que representa o item.
            rating_column: Nome da coluna que representa a avaliação.
            negative_strategy: Estratégia para tratar amostras negativas. Pode ser 'confidence', 'binary', 'discard' ou None.
            hyperparameters: Hiperparâmetros do algoritmo [AlternatingLeastSquares](https://benfred.github.io/implicit/api/models/cpu/als.html#implicit.cpu.als.AlternatingLeastSquares).
        '''
        super().__init__(user_column, item_column, rating_column, hyperparameters)

        if src.EXECUTION_MODE == 'cpu':
            self.recommender = implicit.cpu.als.AlternatingLeastSquares(**self.model_hyperparameters, random_state=src.RANDOM_STATE)
        elif src.EXECUTION_MODE == 'gpu':
            self.recommender = implicit.gpu.als.AlternatingLeastSquares(**self.model_hyperparameters, random_state=src.RANDOM_STATE)
        else:
            raise ValueError('Invalid execution mode. Use "cpu" or "gpu".')
        

    def _get_ratings(self, interactions_df: pd.DataFrame) -> np.ndarray:
        '''
        Retorna as avaliações das interações.

        params:
            interactions_df: DataFrame contendo as interações usuário-item.

        returns:
            Array contendo as avaliações das interações processadas de acordo com a estratégia dos negativos.
        '''
        middle_rating = (interactions_df[self.rating_column].max() - interactions_df[self.rating_column].min()) / 2
        if self.negative_strategy == 'confidence':
            return interactions_df[self.rating_column].values - middle_rating + 1e-15
        elif self.negative_strategy == 'binary':
            return -2 * (interactions_df[self.rating_column]<middle_rating) + 1
        elif self.negative_strategy == 'discard':
            return (interactions_df[self.rating_column]>=middle_rating).astype(int)
        elif self.negative_strategy is None:
            return interactions_df[self.rating_column].values
        else:
            raise ValueError('Invalid negative strategy. Use "confidence", "binary", "exclude" or None.')
        