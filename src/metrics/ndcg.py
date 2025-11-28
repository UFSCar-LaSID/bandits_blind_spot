
import pandas as pd
import numpy as np

import src

def batch_ndcg(predictions_df: pd.DataFrame, **kwargs) -> float:
    '''
    Calcula a métrica NDCG em formato de batch.

    A métrica foi implementada seguindo a definição do livro ["Recommender Systems Handbook"](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_15) (capítulo 8.3.2.3, página 277).

    params:
        predictions_df: é um DataFrame de recomendações feitas para cada interação de teste. Precisa minimamente possuir as seguintes colunas:
            - <COLUMN_USER_ID>: ID do usuário que está recebendo as recomendações.
            - <COLUMN_ITEM_ID>: ID do item que o usuário consumiu de fato na interação de teste.
            - <COLUMN_RECOMMENDATIONS>: uma lista Top-N de IDs de itens recomendados para o usuário naquela interação de teste.
        kwargs: não influencia, apenas colocado por padronização
    
    returns:
        O valor obtido da métrica NDCG.
    '''
    def get_index(x):
        pos = np.asarray(x[src.COLUMN_ITEM_ID]==x[src.COLUMN_RECOMMENDATIONS]).nonzero()[0]
        return pos[0]+1 if len(pos) > 0 else np.nan

    final_df = predictions_df[predictions_df[src.COLUMN_IS_POSITIVE]]

    dcg = (1/np.log2(final_df.apply(get_index, axis=1)+1)).sum(skipna=True)

    idcg_vals = 1/np.log2(np.arange(src.TOP_N)+2)
    idcg = final_df.groupby(src.COLUMN_USER_ID).size().clip(upper=src.TOP_N).apply(lambda x: idcg_vals[:x].sum()).sum()

    return dcg / idcg
    

def ndcg(predictions_df: pd.DataFrame, **kwargs) -> float:
    '''
    Calcula a métrica NDCG.

    A métrica foi implementada seguindo a definição do livro ["Recommender Systems Handbook"](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_15) (capítulo 8.3.2.3, página 277).

    params:
        predictions_df: é um DataFrame de recomendações feitas para cada interação de teste. Precisa minimamente possuir as seguintes colunas:
            - <COLUMN_USER_ID>: ID do usuário que está recebendo as recomendações.
            - <COLUMN_ITEM_ID>: ID do item que o usuário consumiu de fato na interação de teste.
            - <COUMN_RATING>: rating do usuário sobre aquele item
            - <COLUMN_RECOMMENDATIONS>: uma lista Top-N de IDs de itens recomendados para o usuário naquela interação de teste.
        kwargs: não influencia, apenas colocado por padronização
    
    returns:
        O valor obtido da métrica NDCG.
    '''
    final_df = predictions_df[predictions_df[src.COLUMN_IS_POSITIVE]]

    idcg = final_df.shape[0]
    
    recs_ranks = np.where(final_df[src.COLUMN_ITEM_ID].values.reshape(-1, 1) == np.array(final_df[src.COLUMN_RECOMMENDATIONS].tolist()))[1]
    dcg = np.sum(1 / np.log2(recs_ranks + 2))

    return dcg / idcg
