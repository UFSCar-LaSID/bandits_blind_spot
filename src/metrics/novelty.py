
import pandas as pd
import numpy as np

import src
from src.datasets.Dataset import Dataset


def novelty(predictions_df: pd.DataFrame, **kwargs) -> float:
    '''
    Calcula a métrica *novelty* (novidade).

    A métrica foi implementada seguindo a definição do livro ["Recommender Systems Handbook"](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_15) (capítulo 8.3.6, página 285).

    params:
        predictions_df: é um DataFrame de recomendações feitas para cada interação de teste. Precisa minimamente possuir as seguintes colunas:
            - <COLUMN_USER_ID>: ID do usuário que está recebendo as recomendações.
            - <COLUMN_ITEM_ID>: ID do item que o usuário consumiu de fato na interação de teste.
            - <COLUMN_RECOMMENDATIONS>: uma lista Top-N de IDs de itens recomendados para o usuário naquela interação de teste.
        kwargs:
            dataset (Dataset): possui o objeto do dataset testado.
            
    returns:
        O valor obtido da métrica *novelty* (novidade).
    '''
    dataset: Dataset = kwargs['dataset']
    pop = np.bincount(dataset._df_interactions[src.COLUMN_ITEM_ID].values) / len(dataset._df_interactions)
    recs = np.array(predictions_df[src.COLUMN_RECOMMENDATIONS].tolist())
    recs = dataset.encode_items_ids(recs)
    return np.sum(-np.log2(pop[recs])) / len(predictions_df)