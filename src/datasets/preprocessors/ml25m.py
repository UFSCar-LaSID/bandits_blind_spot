
# Link para download da base original: https://grouplens.org/datasets/movielens/

from datetime import datetime
import os
import pandas as pd
import src

def preprocess_ml25m(input_path: str, output_path: str):
    # ======================================= INTERACOES =======================================
    # Lê CSV de interações
    df_interactions = pd.read_csv(os.path.join(input_path, 'ratings.csv'))
    df_interactions.columns = [src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, src.COLUMN_RATING, src.COLUMN_TIMESTAMP]
    # Gera datetimes
    df_interactions[src.COLUMN_DATETIME] = df_interactions[src.COLUMN_TIMESTAMP].apply(lambda x: str(datetime.fromtimestamp(x)))

    # ========================================= SALVAR =========================================
    # Salva bases
    for df, file_name in [(df_interactions, src.FILE_INTERACTIONS)]:
        df.to_csv(os.path.join(output_path, file_name), sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)