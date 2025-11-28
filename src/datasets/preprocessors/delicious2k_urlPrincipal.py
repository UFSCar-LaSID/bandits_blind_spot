
# Link para download da base original: https://grouplens.org/datasets/hetrec-2011/

from datetime import datetime
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import src

def preprocess_delicious2k_urlPrincipal(input_path: str, output_path: str):
    # Nomes das colunas
    COLUMN_TAG_ID = 'id_tag'
    COLUMN_TITLE = 'title'
    COLUMN_URL = 'url'
    COLUMN_URL_PRINCIPAL = 'urlPrincipal'

    if not os.path.exists(input_path):
        input_path = os.path.join(src.DIR_RAW, 'delicious2k')
    # ================================================================================================
    # ========================================= URL COMPLETA =========================================
    # ================================================================================================

    # ----------------------------------- INTERACTIONS & BOOKMARKS -----------------------------------
    # Lê CSV de tags
    df_interactions = pd.read_csv(os.path.join(input_path, 'user_taggedbookmarks.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
    df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']] = df_interactions[['year', 'month', 'day', 'hour', 'minute', 'second']].astype(str)
    df_interactions[src.COLUMN_DATETIME] =  df_interactions['year'].str.zfill(4) + '-' + df_interactions['month'].str.zfill(2) + '-' + df_interactions['day'].str.zfill(2) + ' ' + df_interactions['hour'].str.zfill(2) + ':' + df_interactions['minute'].str.zfill(2) + ':' + df_interactions['second'].str.zfill(2)

    # Gera timestamp
    df_interactions[src.COLUMN_TIMESTAMP] = df_interactions[src.COLUMN_DATETIME].apply(lambda x: int(datetime.timestamp(datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))))

    # Lê CSV de bookmarks
    df_bookmarks = pd.read_csv(os.path.join(input_path, 'bookmarks.dat'), sep='\t', header=0, index_col=False, encoding='iso-8859-1')
    df_bookmarks = df_bookmarks.drop(columns=['md5', 'md5Principal'])

    # Padroniza URLs
    le_url_lower = LabelEncoder()
    df_bookmarks[src.COLUMN_ITEM_ID] = le_url_lower.fit_transform(df_bookmarks['url'].str.lower())
    id_dict = df_bookmarks.set_index('id')[src.COLUMN_ITEM_ID].to_dict()

    # Adiciona ID padronizado nas interações
    df_interactions[src.COLUMN_ITEM_ID] = df_interactions['bookmarkID'].map(id_dict)

    # Limpa e formata os dataframes
    df_interactions = df_interactions.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second', 'bookmarkID'])
    df_interactions.columns = [src.COLUMN_USER_ID, COLUMN_TAG_ID, src.COLUMN_DATETIME, src.COLUMN_TIMESTAMP, src.COLUMN_ITEM_ID]
    df_interactions = df_interactions[[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, COLUMN_TAG_ID, src.COLUMN_TIMESTAMP, src.COLUMN_DATETIME]].sort_values(src.COLUMN_DATETIME)

    # Remove duplicatas
    # df_interactions = df_interactions.drop_duplicates(subset=[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID])

    df_bookmarks = df_bookmarks.drop(columns=['id'])
    df_bookmarks.columns = [COLUMN_TITLE, COLUMN_URL, COLUMN_URL_PRINCIPAL, src.COLUMN_ITEM_ID]
    df_bookmarks = df_bookmarks[[src.COLUMN_ITEM_ID, COLUMN_TITLE, COLUMN_URL, COLUMN_URL_PRINCIPAL]].sort_values(src.COLUMN_ITEM_ID)

    # ================================================================================================
    # ======================================== URL PRINCIPAL =========================================
    # ================================================================================================

    # ----------------------------------- INTERACTIONS & BOOKMARKS -----------------------------------
    # Gera um novo ID
    le_url_principal = LabelEncoder()
    df_bookmarks['urlPrincipalID'] = le_url_principal.fit_transform(df_bookmarks[COLUMN_URL_PRINCIPAL].str.lower())
    id_dict = df_bookmarks.set_index(src.COLUMN_ITEM_ID)['urlPrincipalID'].to_dict()
    df_interactions['urlPrincipalID'] = df_interactions[src.COLUMN_ITEM_ID].map(id_dict)

    # Limpa dataframes
    df_bookmarks = df_bookmarks.drop(columns=[src.COLUMN_ITEM_ID, COLUMN_TITLE, COLUMN_URL]).drop_duplicates(keep='first').rename(columns={'urlPrincipalID': src.COLUMN_ITEM_ID})[[src.COLUMN_ITEM_ID, COLUMN_URL_PRINCIPAL]].sort_values(src.COLUMN_ITEM_ID)
    df_interactions = df_interactions.drop(columns=[src.COLUMN_ITEM_ID]).rename(columns={'urlPrincipalID': src.COLUMN_ITEM_ID})[[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, COLUMN_TAG_ID, src.COLUMN_TIMESTAMP, src.COLUMN_DATETIME]].sort_values(src.COLUMN_DATETIME)

    # ------------------------------------------- SALVAR --------------------------------------------
    # Cria pasta de saída
    os.makedirs(output_path, exist_ok=True)

    # Salva bases
    for df, file_name in [(df_interactions, src.FILE_INTERACTIONS)]:
        df.to_csv(os.path.join(output_path, file_name), sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)