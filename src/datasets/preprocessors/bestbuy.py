# Link para download da base original: https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big

from datetime import datetime
import os
import pandas as pd
import src

def preprocess_bestbuy(input_path: str, output_path: str):

    COLUMN_QUERY = 'query'
    COLUMN_QUERY_DATETIME = 'query_datetime'
    COLUMN_QUERY_TIMESTAMP = 'query_timestamp'

    # Monta caminho da pasta de saída
    os.makedirs(output_path, exist_ok=True)

    # Le arquivo de treinamento
    df_interactions = pd.read_csv(os.path.join(input_path, 'train.csv'), sep=',', header=0)

    # Limpa e arruma arquivo de interações
    df_interactions = df_interactions.drop(columns='category')
    df_interactions.columns = [src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, COLUMN_QUERY, src.COLUMN_DATETIME, COLUMN_QUERY_DATETIME]
    # Padroniza formato dos datetimes
    df_interactions[src.COLUMN_DATETIME] = df_interactions[src.COLUMN_DATETIME].apply(lambda x: datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
    df_interactions[COLUMN_QUERY_DATETIME] = df_interactions[COLUMN_QUERY_DATETIME].apply(lambda x: datetime.strptime(x.split('.')[0], '%Y-%m-%d %H:%M:%S'))
    # Gera timestamps
    df_interactions[src.COLUMN_TIMESTAMP] = df_interactions[src.COLUMN_DATETIME].apply(lambda x: int(datetime.timestamp(x)))
    df_interactions[COLUMN_QUERY_TIMESTAMP] = df_interactions[COLUMN_QUERY_DATETIME].apply(lambda x: int(datetime.timestamp(x)))

    # Salva arquivos
    for df, file_name in [(df_interactions, src.FILE_INTERACTIONS)]:
        df.to_csv(os.path.join(output_path, file_name), sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)