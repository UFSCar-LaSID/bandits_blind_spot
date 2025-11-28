
import src
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from unidecode import unidecode
tqdm.pandas()


def preprocess_amazon_books(input_path: str, output_path: str):

    # Monta caminho das pastas
    os.makedirs(output_path, exist_ok=True)

    # Carrega bases de dados
    df_data = pd.read_csv(os.path.join(input_path, 'books_data.csv'), sep=',')
    df_rating = pd.read_csv(os.path.join(input_path, 'Books_rating.csv'), sep=',')

    def preproc_name(x):
        name = x.lower().replace('&', 'and')
        for c in ['.', ',', "'", '"', '*']:
            name = name.replace(c, '')
        for c in ['  ', '   ', '/', ' - ', ' : ', '- ']:
            name = name.replace(c, ' ')
        return name.strip()

    def preproc_category(category):
        return preproc_name(unidecode(literal_eval(category)[0]))

    # Gera dataframe de categorias
    # Limpa dados
    df_data['categories'] = df_data['categories'].fillna("['null_value']").progress_apply(preproc_category).replace('null_value', np.NaN)
    # Codifica
    from sklearn.preprocessing import LabelEncoder
    le_category = LabelEncoder()
    df_data['categories'] = le_category.fit_transform(df_data['categories'])
    df_data['categories'].replace(le_category.transform([np.nan])[0], '')
    # Gera dataframe final


    # Gera dataframe de autores
    # Formata os nomes
    def preproc_author(author):
        author = unidecode(preproc_name(author).replace('(', '').replace(')', ''))
        for c in ['(', ')', '?', 'ʻ', '_', 'ʾ', '(c)', '"', '*', '`', '/', '|']:
            author = author.replace(c, '')
        if author in ['-mk-', '[anonymus ac01401231]', '[anonymus ac02518615]']:
            return ''
        author = author.strip()
        if len(author) > 0 and author[0] == "'":
            author = author[1:]
        author = author.replace('  ', ' ').replace('   ', ' ')
        return author.title()

    def format_authors(authors):
        if authors is np.NaN:
            return np.NaN
        authors = '/'.join([preproc_author(author) for author in literal_eval(authors)])
        if len(authors) == 0:
            return np.NaN
        if authors[0] == '/':
            authors = authors[1:]
        if authors[-1] == '/':
            authors = authors[:-1]
        return authors

    df_data['authors'] = df_data['authors'].progress_apply(format_authors).replace('', np.NaN)


    # Calcula qtde de autores
    df_data['n_authors'] = df_data['authors'].apply(lambda x: (len(x.split('/'))) if x is not np.NaN else 0)

    # Recupera nome de todos autores
    all_authors = set()
    for authors in df_data[df_data['n_authors']>0]['authors']:
        all_authors.update(authors.split('/'))

    # Codifica autores
    le_author = LabelEncoder()
    le_author.fit(list(all_authors))

    # Unico autor
    df_data.loc[df_data['n_authors']==1,'authors'] = le_author.transform(df_data[df_data['n_authors']==1]['authors']).astype(str)

    # Multiplos autores
    for n in tqdm(sorted(df_data['n_authors'].unique())[2:]):
        df_data.loc[df_data['n_authors']==n, 'authors'] = np.apply_along_axis(
            func1d=lambda x: np.array('/'.join(x.astype(str)), dtype='<U654'),
            axis=1,
            arr=le_author.transform(np.array([i for i in df_data[df_data['n_authors']==n]['authors'].apply(lambda x: x.split('/'))]).flatten()).reshape(-1, n)
        )

    # Trata datas
    df_data['publishedDate'] = df_data['publishedDate'].replace('101-01-01', np.NaN).replace('1016-10-11', np.NaN)

    def preproc_date(date):
        if date is np.NaN:
            return np.NaN
        date = date.replace('*', '').replace('?', '0')
        if len(date) == 4:
            date += '-01-01'
        if len(date) == 7:
            date += '-01'
        return date[:10]

    df_data['publishedDate'] = df_data['publishedDate'].progress_apply(preproc_date)

    # Remove livros sem titulo
    df_data = df_data.dropna(subset='Title')

    # Cria IDs para os items
    le_items = LabelEncoder()
    df_data[src.COLUMN_ITEM_ID] = le_items.fit_transform(df_data['Title'])

    # Altera ordem das colunas
    df_data = df_data[[src.COLUMN_ITEM_ID, *df_data.columns[:-1]]]

    # Remove interacoes sem item ou usuario
    df_rating = df_rating.dropna(subset=['Title', 'User_id'])

    # Gera df de usuario
    df_rating.loc[:, 'profileName'] = df_rating['profileName'].apply(lambda x: str(x).replace('"', '').replace(';', '').replace("'", ''))
    df_users = df_rating[['User_id', 'profileName']].drop_duplicates(subset='User_id', keep='first').copy()
    df_users = df_users.rename(columns={'User_id': src.COLUMN_USER_ID})

    # Renomeia colunas
    df_rating = df_rating.rename(columns={
        'User_id': src.COLUMN_USER_ID,
        'review/helpfulness': 'helpfulness', 
        'review/score': src.COLUMN_RATING, 
        'review/time': src.COLUMN_TIMESTAMP, 
        'review/summary': 'summary',
        'review/text': 'text'
    })

    # Converte helpfulness para taxa
    def preproc_helpfulness(rate):
        if rate is np.NaN or rate == '0/0':
            return np.NaN
        return int(rate.split('/')[0]) / int(rate.split('/')[1])
        
    df_rating.loc[:, 'helpfulness'] = df_rating['helpfulness'].apply(preproc_helpfulness)

    # Adiciona ID dos itens
    df_rating[src.COLUMN_ITEM_ID] = le_items.transform(df_rating['Title'])

    # Remove colunas desnecessarias
    df_rating = df_rating.drop(columns=['Id', 'Title', 'profileName'])

    # Remove interacoes sem timestamp
    df_rating = df_rating[df_rating[src.COLUMN_TIMESTAMP]>0]

    # Gera datetime
    from datetime import datetime
    df_rating.loc[:, src.COLUMN_DATETIME] = df_rating[src.COLUMN_TIMESTAMP].apply(lambda x: str(datetime.fromtimestamp(x)))

    # Reordena colunas
    df_rating = df_rating[[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, src.COLUMN_RATING, src.COLUMN_DATETIME, src.COLUMN_TIMESTAMP, 'Price', 'helpfulness', 'summary', 'text']]

    # Gera df de interacoes
    df_interactions = df_rating.copy()

    def format_column_name(col):
        return ''.join([l.lower() if not l.isupper() or i == 0 else ('_' + l.lower()) for i, l in enumerate(col)])

    # Formata nome de colunas e salva arquivos
    for df, file_name in [(df_interactions, src.FILE_INTERACTIONS)]:
        df.columns = [format_column_name(col) for col in df.columns]
        df.to_csv(os.path.join(output_path, file_name), sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, header=True, index=False)