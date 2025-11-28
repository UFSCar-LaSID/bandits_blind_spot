
import pandas as pd
import numpy as np
import src


def coverage(predictions_df: pd.DataFrame, **kwargs) -> float:
    '''
    Calcula a métrica *coverage* (cobertura).

    A métrica foi implementada seguindo a definição do livro ["Recommender Systems Handbook"](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_15) (capítulo 8.3.3.1, página 282).

    params:
        predictions_df: é um DataFrame de recomendações feitas para cada interação de teste. Precisa minimamente possuir as seguintes colunas:
            - <COLUMN_USER_ID>: ID do usuário que está recebendo as recomendações.
            - <COLUMN_ITEM_ID>: ID do item que o usuário consumiu de fato na interação de teste.
            - <COLUMN_RECOMMENDATIONS>: uma lista Top-N de IDs de itens recomendados para o usuário naquela interação de teste.
        kwargs: não influencia, apenas colocado por padronização
    
    returns:
        O valor obtido da métrica *coverage* (cobertura).
    '''
    items_unique = predictions_df[src.COLUMN_ITEM_ID].unique()
    recs_unique = np.unique(np.array(predictions_df[src.COLUMN_RECOMMENDATIONS].tolist()))

    return len(recs_unique) / len(items_unique)
