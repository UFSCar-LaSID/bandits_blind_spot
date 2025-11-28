# Link para download da base original: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

from datetime import datetime
import os
import pandas as pd
import src

def preprocess_retailrocket(input_path: str, output_path: str):

    COLUMN_INTERACTION_TYPE = 'type'
    COLUMN_TRANSACTION_ID = 'id_transaction'


    # ========================================= INTERAÇÕES =========================================
    # Lê CSV de interações
    df_interactions = pd.read_csv(os.path.join(input_path, 'events.csv'), sep=',', header=0, index_col=False)

    # Arruma nome das colunas
    df_interactions.columns = ['wrong_timestamp', src.COLUMN_USER_ID, COLUMN_INTERACTION_TYPE, src.COLUMN_ITEM_ID, COLUMN_TRANSACTION_ID]

    # Padroniza timestamps de itens da mesma interação
    last_timestamps = df_interactions.groupby(COLUMN_TRANSACTION_ID)['wrong_timestamp'].max()
    df_interactions[src.COLUMN_TIMESTAMP] = df_interactions[COLUMN_TRANSACTION_ID].map(last_timestamps.to_dict()).fillna(df_interactions['wrong_timestamp']).astype(int)
    df_interactions.drop(columns='wrong_timestamp')
    df_interactions = df_interactions.drop_duplicates(keep='first')

    # Modifica o timestamp para segundos ao invés de milisegundos (padrão das outras bases de dados)
    #df_interactions[src.COLUMN_TIMESTAMP] = df_interactions[src.COLUMN_TIMESTAMP] // 1000

    # Gera datetime
    df_interactions[src.COLUMN_DATETIME] = (df_interactions[src.COLUMN_TIMESTAMP] // 1000).apply(lambda x: str(datetime.fromtimestamp(x)).split('.')[0])

    # Reordena colunas
    df_interactions = df_interactions[[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, COLUMN_INTERACTION_TYPE, COLUMN_TRANSACTION_ID, src.COLUMN_TIMESTAMP, src.COLUMN_DATETIME]]
    df_interactions = df_interactions.sort_values([src.COLUMN_TIMESTAMP, src.COLUMN_USER_ID, src.COLUMN_ITEM_ID])
    df_interactions[src.COLUMN_TIMESTAMP] = df_interactions[src.COLUMN_TIMESTAMP] // 1000

    

    # =========================================== SALVAR ===========================================
    os.makedirs(output_path, exist_ok=True)

    # Salva bases
    df_interactions.to_csv(os.path.join(output_path, src.FILE_INTERACTIONS), sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)