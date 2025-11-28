
import pandas as pd
import numpy as np
import src


def hit_rate(predictions_df: pd.DataFrame, **kwargs) -> float:
    '''
    Calcula a métrica [*hit rate* (hr)](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems#hit-rate)

    params:
        predictions_df: é um DataFrame de recomendações feitas para cada interação de teste. Precisa minimamente possuir as seguintes colunas:
            - <COLUMN_USER_ID>: ID do usuário que está recebendo as recomendações.
            - <COLUMN_ITEM_ID>: ID do item que o usuário consumiu de fato na interação de teste.
            - <COUMN_RATING>: rating do usuário sobre aquele item
            - <COLUMN_RECOMMENDATIONS>: uma lista Top-N de IDs de itens recomendados para o usuário naquela interação de teste.
        kwargs: não influencia, apenas colocado por padronização
    
    returns:
        O valor obtido da métrica *hit rate* (hr)
    '''
    final_df = predictions_df[predictions_df[src.COLUMN_IS_POSITIVE]]
    return (final_df[src.COLUMN_ITEM_ID].values.reshape(-1, 1) == np.array(final_df[src.COLUMN_RECOMMENDATIONS].tolist())).any(axis=1).mean()