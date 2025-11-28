
import src
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def preprocess_amazon_games(input_path: str, output_path: str):

    os.makedirs(output_path, exist_ok=True)

    # Carrega bases de dados
    import json

    # Interacoes
    with open(os.path.join(input_path, 'Video_Games.jsonl'), 'r') as fp:
        reviews = fp.readlines()
    for i, line in tqdm(enumerate(reviews), total=len(reviews)):
        reviews[i] = json.loads(reviews[i].strip())
    df_interactions = pd.DataFrame.from_records(reviews)

    # Processa df de interacoes
    # Renomeia colunas do dataframe
    df_interactions = df_interactions.reset_index().rename(
        columns={'index': 'id_review', 'rating': src.COLUMN_RATING, 'parent_asin': src.COLUMN_ITEM_ID,
                'asin': 'id_item_style', 'user_id': src.COLUMN_USER_ID, 'timestamp': src.COLUMN_TIMESTAMP}
    )

    # Cria dataframe de imagens
    df_interactions['images_count'] = df_interactions['images'].progress_apply(len)
    df_images_reviews = list()
    reviews_with_image = df_interactions[df_interactions['images_count']>0].copy()
    for _, row in tqdm(reviews_with_image.iterrows(), total=len(reviews_with_image)):
        for image_url in row['images']:
            image_url['id_review'] = row['id_review']
            df_images_reviews.append(image_url)
    del reviews_with_image
    df_images_reviews = pd.DataFrame.from_records(df_images_reviews).drop(columns=['attachment_type'])
    df_images_reviews = df_images_reviews[['id_review']+list(df_images_reviews.columns)[:-1]]
    df_interactions.drop(columns=['images_count', 'images'], inplace=True)

    # Arruma formato das colunas
    df_interactions[src.COLUMN_RATING] = df_interactions[src.COLUMN_RATING].astype(int)
    df_interactions[src.COLUMN_ITEM_ID] = df_interactions[src.COLUMN_ITEM_ID].astype(str)
    df_interactions['id_item_style'] = df_interactions['id_item_style'].astype(str)
    df_interactions[src.COLUMN_USER_ID] = df_interactions[src.COLUMN_USER_ID].astype(str)
    df_interactions['verified_purchase'] = df_interactions['verified_purchase'].astype(int)

    # Gera coluna de datetime
    from datetime import datetime
    df_interactions[src.COLUMN_TIMESTAMP] = df_interactions[src.COLUMN_TIMESTAMP]//1000
    df_interactions[src.COLUMN_DATETIME] = df_interactions[src.COLUMN_TIMESTAMP].progress_apply(datetime.fromtimestamp)

    # Ordena colunas do df interacoes
    main_cols = [src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, src.COLUMN_RATING, src.COLUMN_DATETIME, src.COLUMN_TIMESTAMP]
    secondary_cols = [col for col in df_interactions.columns if col not in main_cols]
    df_interactions = df_interactions[main_cols+secondary_cols]

    # Salva arquivos
    for df, file_name in tqdm([(df_interactions, src.FILE_INTERACTIONS)]):
        df.to_csv(os.path.join(output_path, file_name), sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)