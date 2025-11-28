

import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_path)

import itertools
import pandas as pd

import src
from src.scripts.utils.parameters_handle import get_input
from src.datasets.Dataset import Dataset
from src.scripts.utils.EmbeddingsLoader import EmbeddingsLoader
from src.scripts.utils.Logger import Logger
from src.scripts.utils.get_save_path import generate_incremental_save_path, get_last_not_incremental_save_path
from src.scripts.incremental.experiment import run_experiment

from src.algorithms.incremental import INCREMENTAL_ALGORITHMS_TABLE
from src.datasets import DATASETS_TABLE
from src.algorithms.not_incremental import NOT_INCREMENTAL_ALGORITHMS_TABLE
from src.algorithms.incremental.contexts_builder import CONTEXTS_BUILDER_TABLE

import time



def main():
    '''
    Código responsável por executar um conjunto de experimentos incremental. 

    O seguintes inputs serão solicitados ao usuário:

    1. Algoritmo incremental (e.g. LinUCB, LinGreedy, LinTemporal)
    2. Base de dados
    3. Algoritmo das embeddings (e.g. ALS e BPR)
    4. Função de geração de contextos (e.g. users embeddings, items mean, items concat, etc.)

    Se mais de uma opção for selecionada em um item, todas as possíveis combinações serão executadas.
    O input pode ser coletado via linha de comando (python input()) ou passado como argumento na execução do script (--setup <setup_file>).
    O arquivo de setup deve ser um arquivo JSON com a seguinte estrutura:

    {
        "datasets": [1, 2],    // Lista de datasets a serem utilizados (IDs de datasets - disponíveis em src/datasets/__init__.py)
        "algorithms": [2, 3],   // Lista de algoritmos a serem utilizados (IDs de algoritmos - disponíveis em src/algorithms/incremental/__init__.py)
        "embeddings": [1, 2],   // Lista de algoritmos de embeddings a serem utilizados (IDs de algoritmos - disponíveis em src/not_incremental/embeddings/__init__.py)
        "contexts": [1, 2],     // Lista de funções de geração de contextos a serem utilizadas (IDs de funções - disponíveis em src/algorithms/incremental/contexts_builder/__init__.py)
    }
    

    Com os inputs coletados, os experimentos serão executados, chamando a função run_experiment do módulo experiment para cada combinação de dataset, algoritmo, gerador de embeddings e construtor de contexto.

    Os resultados dos experimentos devem ser salvos no caminho: <src.DIR_EXPERIMENTS>/<dataset_name>/incremental/<incremental_algo_name>/<embeddings_algo_name>/<build_context_name>/<timestamp_atual>.
    '''
    
    incremental_algo_options, datasets_options, embeddings_options, contexts_options = get_input(
        'Selecione as opções para rodar experimentos incrementais (MAB)',
        [
            {
                'name': 'algorithms',
                'description': 'Algoritmos incrementais (MAB) que serão treinados e avaliados',
                'name_column': src.INCREMENTAL_ALGORITHM_NAME,
                'options': INCREMENTAL_ALGORITHMS_TABLE
            },
            {
                'name': 'datasets',
                'description': 'Datasets que serão usados para treinamento/validação/teste',
                'name_column': src.DATASET_NAME,
                'options': DATASETS_TABLE
            },
            {
                'name': 'embeddings',
                'description': 'Embeddings de quais algoritmos serão usadas para gerar os contextos',
                'name_column': src.NOT_INCREMENTAL_ALGORITHM_NAME,
                'options': NOT_INCREMENTAL_ALGORITHMS_TABLE
            },
            {
                'name': 'contexts',
                'description': 'Quais tipos de contextos serão usados como entrada dos algoritmos incrementais',
                'name_column': src.CONTEXTS_BUILDER_NAME,
                'options': CONTEXTS_BUILDER_TABLE
            }
        ]
    )
    
    start_time = time.time()
    algo_options_iter = list(itertools.product(incremental_algo_options, embeddings_options, contexts_options))

    for dataset_option in datasets_options:
        dataset_name = DATASETS_TABLE.loc[dataset_option, src.DATASET_NAME]
        print(f'Loading dataset {dataset_name}...')
        dataset = Dataset(dataset_name, load_timediff=False)
        print(f'Dataset {dataset_name} loaded...')

        for incremental_algo_option, embeddings_option, context_option in algo_options_iter:
            Incremental_algo = INCREMENTAL_ALGORITHMS_TABLE.loc[incremental_algo_option, src.INCREMENTAL_ALGORITHM_CLASS]
            incremental_name = INCREMENTAL_ALGORITHMS_TABLE.loc[incremental_algo_option, src.INCREMENTAL_ALGORITHM_NAME]
            incremental_hyperparams = INCREMENTAL_ALGORITHMS_TABLE.loc[incremental_algo_option, src.INCREMENTAL_ALGORITHM_HYPERPARAMETERS]

            Embedding_generator = NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[embeddings_option, src.NOT_INCREMENTAL_ALGORITHM_CLASS]
            embeddings_name = NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[embeddings_option, src.NOT_INCREMENTAL_ALGORITHM_NAME]

            ContextBuilder = CONTEXTS_BUILDER_TABLE.loc[context_option, src.CONTEXTS_BUILDER_CLASS]
            contexts_name = CONTEXTS_BUILDER_TABLE.loc[context_option, src.CONTEXTS_BUILDER_NAME]
            contexts_hyperparams = CONTEXTS_BUILDER_TABLE.loc[context_option, src.CONTEXTS_BUILDER_HYPERPARAMETERS]

            embeddings_load_path = get_last_not_incremental_save_path(dataset_name, embeddings_name)
            experiment_save_path = generate_incremental_save_path(dataset_name, incremental_name, embeddings_name, contexts_name)

            logger = Logger(experiment_save_path)

            embeddings_loader = EmbeddingsLoader(Embedding_generator, embeddings_load_path, dataset)
            best_embeddings_hyperparameters = embeddings_loader.get_best_hyperparameters(add_params_to_list=True)

            run_experiment(
                Algorithm=Incremental_algo,
                dataset=dataset,
                logger=logger,
                embeddings_loader=embeddings_loader,
                ContextsBuilder=ContextBuilder,
                save_path=experiment_save_path,
                incremental_algo_grid_search_params=incremental_hyperparams,
                embeddings_grid_search_params=best_embeddings_hyperparameters,
                context_builder_grid_search_params=contexts_hyperparams
            )

    print(f'Demorou {time.time() - start_time} segundos')
    print('Experiments finished successfully !')



if __name__ == '__main__':
    main()
    sys.exit(0)