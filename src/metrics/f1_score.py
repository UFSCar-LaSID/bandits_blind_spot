
import pandas as pd
import numpy as np
import src



def f1_score(predictions_df: pd.DataFrame, **kwargs) -> float:
    '''
    Calcula a métrica *f1_score* (f-medida).

    A métrica foi implementada seguindo a definição do livro ["Recommender Systems Handbook"](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_15) (capítulo 8.3.2.2, página 274 / capítulo 19.4.4, página 637).

    params:
        predictions_df: é um DataFrame de recomendações feitas para cada interação de teste. Precisa minimamente possuir as seguintes colunas:
            - <COLUMN_USER_ID>: ID do usuário que está recebendo as recomendações.
            - <COLUMN_ITEM_ID>: ID do item que o usuário consumiu de fato na interação de teste.
            - <COUMN_RATING>: rating do usuário sobre aquele item
            - <COLUMN_RECOMMENDATIONS>: uma lista Top-N de IDs de itens recomendados para o usuário naquela interação de teste.
        kwargs: não influencia, apenas colocado por padronização
    
    returns:
        O valor obtido da métrica *f1_score* (f-medida).
    '''
    final_df = predictions_df[predictions_df[src.COLUMN_IS_POSITIVE]]
    recs = np.array(final_df[src.COLUMN_RECOMMENDATIONS].tolist())

    hits = (final_df[src.COLUMN_ITEM_ID].values.reshape(-1, 1) == recs).any(axis=1).sum()
    precision = hits / (len(final_df) * recs.shape[1])
    recall = hits / (len(final_df))

    if precision == 0 and recall == 0:
        return 0

    return 2 * ((precision * recall) / (precision + recall))
