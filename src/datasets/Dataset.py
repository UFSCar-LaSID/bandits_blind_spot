import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Iterator, Union

import src
from src.datasets import DATASETS_TABLE

class Dataset:
    '''
    Classe que representa um dataset de interações usuário-item. 
    Possui métodos para carregar as interações nos diversos splits (treino, validação e janelas de teste) e decodificar os id's de usuários e itens.

    Mais informações sobre o Dataset pode ser consultado no [README](https://github.com/RecSys-UFSCar/mab-recsys/tree/main/src/datasets/README.md)
    '''

    def __init__(self, name: str, remove_cold_start: bool = True, remove_recurring_interactions: bool = True, missing_ratings: str = 'drop', load_timediff: bool = False):
        '''
        Construtor da classe Dataset. 
        
        Na inicialização, é necessário fornecer um nome para o dataset. Com esse nome, será buscado por suas interações no seguinte caminho: `<DIR_DATASET>/<name>/<FILE_INTERACTIONS>`. A inicialização também carrega as interações para a memória, codificando os id's e realizando splits.

        params:
            name: Nome do dataset a ser carregado
            n_test_splits: Quantidade de splits da particao de teste
            remove_cold_start: Remocao de cold start para itens
            remove_recurring_interactions: Remocao de interacoes recorrentes 
        '''
        # Store dataset name
        self._name = name

        # Get dataset row from DATASETS_TABLE
        if name not in DATASETS_TABLE[src.DATASET_NAME].values:
            raise ValueError(f"Dataset not found: {name}")
        dataset_info = DATASETS_TABLE[DATASETS_TABLE[src.DATASET_NAME] == name].iloc[0]
        
        # Load interactions
        files_dir = os.path.join(src.DIR_DATASET, name)
        filepath_interactions = os.path.join(files_dir, src.FILE_INTERACTIONS)
        if not os.path.exists(filepath_interactions):
            raise FileNotFoundError(f"Interactions file not found: {filepath_interactions}")
        self._df_interactions = pd.read_csv(filepath_interactions, delimiter=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR)
        if src.COLUMN_DATETIME in self._df_interactions.columns:
            self._df_interactions[src.COLUMN_DATETIME] = pd.to_datetime(self._df_interactions[src.COLUMN_DATETIME], format='%Y-%m-%d %H:%M:%S')
        
        print('interacoes')
        print(self._df_interactions.shape[0])

        print('items')
        print(self._df_interactions[src.COLUMN_ITEM_ID].nunique())

        print('users')
        print(self._df_interactions[src.COLUMN_USER_ID].nunique())
        
        # Handle ratings
        if src.COLUMN_RATING in self._df_interactions.columns:
            self._df_interactions = dataset_info[src.DATASET_MISSING_RATINGS_FUNC](self._df_interactions)
        else:
            # Create placeholder ratings for implicit datasets
            self._df_interactions[src.COLUMN_RATING] = 1
        
        if src.COLUMN_DATETIME in self._df_interactions.columns:
            self._df_interactions = self._df_interactions[[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, src.COLUMN_RATING, src.COLUMN_TIMESTAMP, src.COLUMN_DATETIME]]
        else:
            self._df_interactions = self._df_interactions[[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID, src.COLUMN_RATING]]
        
        # Define positive interactions
        self._df_interactions[src.COLUMN_IS_POSITIVE] = dataset_info[src.DATASET_IS_POSITIVE_FUNC](self._df_interactions)

        # Sort interactions by timestamp
        if src.COLUMN_DATETIME in self._df_interactions.columns:
            self._df_interactions.sort_values(by=[src.COLUMN_TIMESTAMP], inplace=True)
        else:
            self._df_interactions[src.COLUMN_DATETIME] = 0
            self._df_interactions[src.COLUMN_TIMESTAMP] = 0
            self._df_interactions[src.COLUMN_DATETIME] = pd.to_datetime(self._df_interactions[src.COLUMN_DATETIME])

        # Treat recurring interactions
        if remove_recurring_interactions:
            self._df_interactions = self._remove_recurring_interactions(self._df_interactions)

        # Create timediff column
        if load_timediff:
            self._df_interactions = self._create_timediff(self._df_interactions)

        # Encode interactions
        self._encode_interactions()
        
        # Split interactions
        self._df_full_train, self._df_train, self._df_val, self._df_test = self._split_interactions(self._df_interactions, remove_cold_start)

    
    def _remove_recurring_interactions(self, df_interactions) -> pd.DataFrame:
        '''
        Remove interações recorrentes do dataset.
        '''
        return df_interactions.drop_duplicates(subset=[src.COLUMN_USER_ID, src.COLUMN_ITEM_ID], keep='first')
    

    def _split_interactions(self, df_interactions: pd.DataFrame, remove_cold_start: bool) -> 'list[pd.DataFrame]':
        '''
        Realiza a divisão das interações em treino, validação e janelas de teste.
        '''
        # Separar treino e teste
        df_full_train, df_test = train_test_split(df_interactions, test_size=src.TEST_SIZE, shuffle=False)
        if remove_cold_start:
            df_test = self._remove_cold_start(df_test, source_df=df_full_train)
        # Separar treino e validacao
        window_size = len(df_test) // src.TEST_WINDOWS_QNT
        df_train, df_val = train_test_split(df_full_train, test_size=window_size, shuffle=False)
        if remove_cold_start:
            df_val = self._remove_cold_start(df_val, source_df=df_train)
        return [df_full_train, df_train, df_val, df_test]


    def _remove_cold_start(self, df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Remove cold start do dataset.
        '''
        item_mask = df[src.COLUMN_ITEM_ID].isin(source_df[src.COLUMN_ITEM_ID].unique())
        user_mask = df[src.COLUMN_USER_ID].isin(source_df[src.COLUMN_USER_ID].unique())
        return df[(item_mask) & (user_mask)]
    

    def _create_timediff(self, df_interactions: pd.DataFrame) -> pd.DataFrame:
        '''
        Cria a coluna com o intervalo temporal
        '''
        # Ordenar o dataframe por usuário e timestamp para garantir que as interações estejam na ordem correta
        df_interactions = df_interactions.sort_values(by=[src.COLUMN_USER_ID, src.COLUMN_DATETIME])

        # Agrupar por usuário e calcular a diferença de tempo (em segundos) entre as interações
        df_interactions[src.COLUMN_TIMEDIFF] = df_interactions.groupby(src.COLUMN_USER_ID)[src.COLUMN_DATETIME].diff().dt.total_seconds().fillna(0)

        def replace_zero_with_previous_timediff(group):
            # Itera sobre o grupo do usuário
            last_valid_timediff = None  # Variável para armazenar o último valor válido (não-nulo)
            for i in range(len(group)):
                if group.iloc[i][src.COLUMN_TIMEDIFF] <= 1e-15:
                    # Se o valor de timediff for 0, substitui pelo último valor válido, se houver
                    if last_valid_timediff is not None:
                        group.at[group.index[i], src.COLUMN_TIMEDIFF] = last_valid_timediff
                else:
                    # Atualiza o último valor válido (não-nulo)
                    last_valid_timediff = group.iloc[i][src.COLUMN_TIMEDIFF]
            return group

        # Aplicar a função a cada grupo de usuários e remover o agrupamento
        df_interactions = df_interactions.groupby(src.COLUMN_USER_ID).apply(replace_zero_with_previous_timediff).reset_index(drop=True)
        return df_interactions
    

    def _encode_interactions(self):
        '''
        Codifica os id's de usuários e itens
        '''
        self._df_interactions[src.COLUMN_USER_ID], self.user_decoder = pd.factorize(self._df_interactions[src.COLUMN_USER_ID])
        self._df_interactions[src.COLUMN_ITEM_ID], self.item_decoder = pd.factorize(self._df_interactions[src.COLUMN_ITEM_ID])

        self.user_decoder = self.user_decoder.to_numpy()
        self.item_decoder = self.item_decoder.to_numpy()

        self.user_encoder = {user_id: idx for idx, user_id in enumerate(self.user_decoder)}
        self.item_encoder = {item_id: idx for idx, item_id in enumerate(self.item_decoder)}


    @property
    def name(self) -> str:
        '''
        Nome do dataset carregado
        '''
        return self._name


    @property
    def train_interactions(self) -> pd.DataFrame:
        '''
        DataFrame contendo as interações de treino **com o split de validação REMOVIDO**
        '''
        return self._df_train
    

    @property
    def val_interactions(self) -> pd.DataFrame:
        '''
        DataFrame contendo as interações de validação
        '''
        return self._df_val

    
    @property
    def test_interactions(self) -> pd.DataFrame:
        '''
        DataFrame contendo as interações de teste
        '''
        return self._df_test


    @property
    def full_train_interactions(self) -> pd.DataFrame:
        '''
        DataFrame contendo **TODAS** as interações de treino (**train + val**)
        '''
        return self._df_full_train


    @property
    def test_splits(self) -> Iterator[pd.DataFrame]:
        '''
        Iterador que retorna um split/janela de teste por vez. Cada split é um DataFrame contendo as interações de teste naquele split. São geradas `TEST_WINDOWS_QNT` janelas de teste.

        Exemplo de uso:
        ```python
        for test_split in dataset.test_splits:
            print(test_split)
        ```
        '''
        window_size = len(self._df_test) // src.TEST_WINDOWS_QNT
        extra_interactions = len(self._df_test) % src.TEST_WINDOWS_QNT
        for window in range(src.TEST_WINDOWS_QNT):
            start = window * window_size
            end = start + window_size + 1 * (window < extra_interactions)
            yield self._df_test.iloc[start:end]


    @property
    def num_interactions(self) -> int:
        '''
        Número de interações no dataset
        '''
        return self._df_interactions.shape[0]


    @property
    def num_users(self) -> int:
        '''
        Número de usuários no dataset
        '''
        return self._df_interactions[src.COLUMN_USER_ID].nunique()


    @property
    def num_items(self) -> int:
        '''
        Número de itens no dataset
        '''
        return self._df_interactions[src.COLUMN_ITEM_ID].nunique()
    

    def encode_users_ids(self, users_ids: 'Union[list[str], pd.Series, np.ndarray]') -> 'Union[list[int], pd.Series, np.ndarray]':
        '''
        Converte uma lista de id's de usuários para seus respectivos id's codificados.

        params:
            users_ids: Lista de id's de usuários a serem codificados.
        
        return:
            Lista de id's de usuários codificados.
        '''
        return np.vectorize(self.user_encoder.get, otypes=[int])(users_ids)


    def decode_users_ids(self, users_ids: 'Union[list[int], pd.Series, np.ndarray]') -> 'Union[list[str], pd.Series, np.ndarray]':
        '''
        Converte uma lista de id's de usuários para seus respectivos id's decodificados (originais que estavam no dataset).

        params:
            users_ids: Lista de id's de usuários a serem decodificados.
        
        return:
            Lista de id's de usuários decodificados.
        '''
        return self.user_decoder[users_ids].tolist()
    
    
    def encode_items_ids(self, items_ids: 'Union[list[str], pd.Series, np.ndarray]') -> 'Union[list[int], pd.Series, np.ndarray]':
        '''
        Converte uma lista de id's de itens para seus respectivos id's codificados.

        params:
            items_ids: Lista de id's de itens a serem codificados.
        
        return:
            Lista de id's de itens codificados.
        '''
        return np.vectorize(self.item_encoder.get, otypes=[int])(items_ids)


    def decode_items_ids(self, items_ids: 'Union[list[int], pd.Series, np.ndarray]') -> 'Union[list[str], pd.Series, np.ndarray]':
        '''
        Converte uma lista de id's de itens para seus respectivos id's decodificados (originais que estavam no dataset).

        params:
            items_ids: Lista de id's de itens a serem decodificados.
        
        return:
            Lista de id's de itens decodificados.
        '''
        return self.item_decoder[items_ids].tolist()
