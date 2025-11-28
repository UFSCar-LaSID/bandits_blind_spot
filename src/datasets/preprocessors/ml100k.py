# Link para download da base original: https://grouplens.org/datasets/movielens/

import src
import os
import pandas as pd
from datetime import datetime


def preprocess_ml100k(input_path: str, output_path: str):

    # Nomes de arquivos
    FILE_INTERACTIONS = 'interactions.csv'

    # Monta caminho das pastas
    os.makedirs(output_path, exist_ok=True)


    # Carrega bases de dados
    df_interactions = pd.read_csv(os.path.join(input_path, 'u.data'), sep='\t', encoding='latin-1', header=None)
    df_interactions.columns = [src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, src.COLUMN_RATING, src.COLUMN_TIMESTAMP]
    df_interactions[src.COLUMN_DATETIME] = df_interactions[src.COLUMN_TIMESTAMP].apply(lambda x: datetime.fromtimestamp(x))
    df_interactions = df_interactions[[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, src.COLUMN_RATING, src.COLUMN_DATETIME, src.COLUMN_TIMESTAMP]]
    df_interactions = df_interactions.sort_values([src.COLUMN_DATETIME, src.COLUMN_USER_ID, src.COLUMN_ITEM_ID])

    # Salva arquivos
    for df, file_name in [(df_interactions, FILE_INTERACTIONS)]:
        df.to_csv(os.path.join(output_path, file_name), sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)