

import src
import os
import json
from src.algorithms.not_incremental.implicit.ImplicitRecommender import ImplicitRecommender
from src.datasets.Dataset import Dataset
from typing import Type
import numpy as np


class EmbeddingsLoader:
    '''
    Classe responsável por carregar as embeddings (lendo do sistema de arquivos as embeddings, e se não existir, é gerado as embeddings).
    '''

    def __init__(self, Embeddings_algo: Type[ImplicitRecommender], algo_results_save_path: str, dataset: Dataset):
        '''
        Construtor da classe EmbeddingsLoader.

        params:
            embeddings_algo: Algoritmo que gera as embeddings.
            algo_results_save_path: Caminho onde os resultados dos experimentos do algoritmo estão salvos.
            dataset: Dataset a ser utilizado para gerar as embeddings caso seja necessário.
        '''
        self.Embeddings_algo = Embeddings_algo
        self.algo_results_save_path = algo_results_save_path
        self.dataset = dataset

        with open(os.path.join(self.algo_results_save_path, src.FILE_METADATA), 'r') as metadata_file:
            metadata_dict = json.load(metadata_file)
        
        self.possible_hyperparameters: dict = metadata_dict["grid_search"]
        self.best_hyperparameters_key = str(metadata_dict["best_hiperparam_combination_id"])
    
    def get_best_hyperparameters(self, add_params_to_list=False):
        best_hyperparameters = self.possible_hyperparameters[self.best_hyperparameters_key]["params"].copy()
        if add_params_to_list:
            for key, value in best_hyperparameters.items():
                best_hyperparameters[key] = [value]
        return best_hyperparameters
    

    def _get_hyperparameters_key(self, hyperparameters: dict) -> str:
        for key, value in self.possible_hyperparameters.items():
            if value["params"] == hyperparameters:
                return key
        
        raise ValueError("hyperparameters given for embeddings was not tested in the not_incremental experiments")

    def load_train_users_embeddings(self, hyperparameters: dict) -> np.ndarray:
        '''
        Carrega as embeddings dos usuários da partição de treino com os hiperparâmetros fornecidos. Caso as embeddings não existam, elas são geradas e salvas no sistema de arquivos.
          
        **Importante:** a partição de treino não possui os dados de validação. Caso queira carregar as embeddings geradas a partir da partição de treino + validação (ou seja, a partição de treino completa), utilize o método `load_train_full_users_embeddings`.

        params:
            hyperparameters: Hiperparâmetros utilizados para gerar as embeddings. É um dicionário onde a chave é o nome do hiperparâmetro e o valor é o valor do hiperparâmetro.
        '''
        hyperparameters_key = self._get_hyperparameters_key(hyperparameters)
        embeddings_path = os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_EMBEDDINGS, src.FILE_USERS_EMBEDDINGS)

        if not os.path.exists(embeddings_path):
            self._generate_train_embeddings(hyperparameters, hyperparameters_key)
        
        return np.load(embeddings_path)

    def load_train_items_embeddings(self, hyperparameters: dict) -> np.ndarray:
        '''
        Carrega as embeddings dos itens da partição de treino com os hiperparâmetros fornecidos. Caso as embeddings não existam, elas são geradas e salvas no sistema de arquivos.

        **Importante:** a partição de treino não possui os dados de validação. Caso queira carregar as embeddings geradas a partir da partição de treino + validação (ou seja, a partição de treino completa), utilize o método `load_train_full_items_embeddings`.

        params:
            hyperparameters: Hiperparâmetros utilizados para gerar as embeddings. É um dicionário onde a chave é o nome do hiperparâmetro e o valor é o valor do hiperparâmetro.
        '''
        hyperparameters_key = self._get_hyperparameters_key(hyperparameters)
        embeddings_path = os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_EMBEDDINGS, src.FILE_ITEMS_EMBEDDINGS)

        if not os.path.exists(embeddings_path):
            self._generate_train_embeddings(hyperparameters, hyperparameters_key)
        
        return np.load(embeddings_path)

    def load_train_full_users_embeddings(self, hyperparameters: dict) -> np.ndarray:
        '''
        Carrega as embeddings dos usuários da partição de treino completa (treino + validação) com os hiperparâmetros fornecidos. Caso as embeddings não existam, elas são geradas e salvas no sistema de arquivos.

        params:
            hyperparameters: Hiperparâmetros utilizados para gerar as embeddings. É um dicionário onde a chave é o nome do hiperparâmetro e o valor é o valor do hiperparâmetro.
        '''
        hyperparameters_key = self._get_hyperparameters_key(hyperparameters)
        embeddings_path = os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_FULL_EMBEDDINGS, src.FILE_USERS_EMBEDDINGS)

        if not os.path.exists(embeddings_path):
            self._generate_train_full_embeddings(hyperparameters, hyperparameters_key)
        
        return np.load(embeddings_path)

    def load_train_full_items_embeddings(self, hyperparameters: dict) -> np.ndarray:
        '''
        Carrega as embeddings dos itens da partição de treino completa (treino + validação). Caso as embeddings não existam, elas são geradas e salvas no sistema de arquivos.

        params:
            hyperparameters: Hiperparâmetros utilizados para gerar as embeddings. É um dicionário onde a chave é o nome do hiperparâmetro e o valor é o valor do hiperparâmetro.
        '''
        hyperparameters_key = self._get_hyperparameters_key(hyperparameters)
        embeddings_path = os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_FULL_EMBEDDINGS, src.FILE_ITEMS_EMBEDDINGS)

        if not os.path.exists(embeddings_path):
            self._generate_train_full_embeddings(hyperparameters, hyperparameters_key)
        
        return np.load(embeddings_path)


    def _generate_train_embeddings(self, hyperparameters: dict, hyperparameters_key: str):
        '''
        Função interna responsável por gerar as embeddings da partição de treino com os hiperparâmetros fornecidos (essa função deve ser chamada apenas se as embeddings não existirem no sistema de arquivos).

        params:
            hyperparameters: Hiperparâmetros utilizados para gerar as embeddings. É um dicionário onde a chave é o nome do hiperparâmetro e o valor é o valor do hiperparâmetro.
        '''
        embeddings_generator = self.Embeddings_algo(hyperparameters=hyperparameters)
        embeddings_generator.train(self.dataset.train_interactions, self.dataset.num_users, self.dataset.num_items)

        np.save(
            os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_EMBEDDINGS, src.FILE_USERS_EMBEDDINGS),
            embeddings_generator.users_embeddings
        )
        np.save(
            os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_EMBEDDINGS, src.FILE_ITEMS_EMBEDDINGS),
            embeddings_generator.items_embeddings
        )

    def _generate_train_full_embeddings(self, hyperparameters: dict, hyperparameters_key: str):
        '''
        Função interna responsável por gerar as embeddings da partição de treino completa (treino + validação) com os hiperparâmetros fornecidos (essa função deve ser chamada apenas se as embeddings não existirem no sistema de arquivos).

        params:
            hyperparameters: Hiperparâmetros utilizados para gerar as embeddings. É um dicionário onde a chave é o nome do hiperparâmetro e o valor é o valor do hiperparâmetro.
        '''
        embeddings_generator = self.Embeddings_algo(hyperparameters=hyperparameters)
        embeddings_generator.train(self.dataset.full_train_interactions, self.dataset.num_users, self.dataset.num_items)

        np.save(
            os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_FULL_EMBEDDINGS, src.FILE_USERS_EMBEDDINGS),
            embeddings_generator.users_embeddings
        )
        np.save(
            os.path.join(self.algo_results_save_path, src.DIR_VALIDATION, hyperparameters_key, src.DIR_TRAIN_FULL_EMBEDDINGS, src.FILE_ITEMS_EMBEDDINGS),
            embeddings_generator.items_embeddings
        )
