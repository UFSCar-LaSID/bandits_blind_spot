
import pandas as pd
import numpy as np
import src
from sklearn.metrics.pairwise import cosine_distances
from src.datasets.Dataset import Dataset



def diversity(predictions_df: pd.DataFrame, **kwargs) -> float:
    '''
    TODO

    params:
        predictions_df: é um DataFrame de recomendações feitas para cada interação de teste. Precisa minimamente possuir as seguintes colunas:
            - <COLUMN_USER_ID>: ID do usuário que está recebendo as recomendações.
            - <COLUMN_ITEM_ID>: ID do item que o usuário consumiu de fato na interação de teste.
            - <COLUMN_RECOMMENDATIONS>: uma lista Top-N de IDs de itens recomendados para o usuário naquela interação de teste.
        kwargs:
            items_embeddings (np.ndarray): embeddings de itens
    
    returns:
        O valor obtido da métrica *diversidade*
    '''
    items_embeddings = kwargs['items_embeddings']
    dataset: Dataset = kwargs['dataset']
    distances = cosine_distances(items_embeddings)
    diversity = 0

    recs = np.array(predictions_df[src.COLUMN_RECOMMENDATIONS].tolist())
    recs = dataset.encode_items_ids(recs)
    for i in range(recs.shape[0]):
        cur_rec = recs[i]
        diversity += distances[
            np.repeat(cur_rec, recs.shape[1]),
            np.repeat(cur_rec.reshape(1, -1), recs.shape[1], axis=0).flatten()
        ].sum() / recs.shape[0]
    
    return diversity
