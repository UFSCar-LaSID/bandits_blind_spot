
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_path)

import src
from src.scripts.utils.parameters_handle import get_input
from src.datasets.Dataset import Dataset
from src.scripts.utils.Logger import Logger
from src.scripts.utils.get_save_path import generate_not_incremental_save_path
from src.scripts.not_incremental.experiment import run_experiment

from src.algorithms.not_incremental import NOT_INCREMENTAL_ALGORITHMS_TABLE
from src.datasets import DATASETS_TABLE


def main():
    '''
    Código responsável por executar um conjunto de experimentos `not_incremental`. Deve coletar o input do usuário, no caso, são:

    1. Os dataset a serem utilizados.
    2. Os algoritmos (não-incremental) a serem utilizados.

    Se mais de um dataset e algoritmo for selecionado, todas combinações de dataset e algoritmo devem ser executadas.
    O input pode ser coletado via linha de comando (python `input()`) ou passado como argumento na execução do script (`--setup <setup_file>`).
    O arquivo de setup deve ser um arquivo JSON com a seguinte estrutura:

    ```
        {
            "datasets": [1, 2],    // Lista de datasets a serem utilizados (IDs de datasets - disponíveis em src/datasets/__init__.py)
            "algorithms": [2, 3]   // Lista de algoritmos a serem utilizados (IDs de algoritmos - disponíveis em src/algorithms/not_incremental/__init__.py)
        }
    ```
    
    Com os inputs coletados, os experimentos serão executados, chamando a função `run_experiment` do módulo `experiment` para cada combinação de dataset e algoritmo.

    Os resultados dos experimentos devem ser salvos no caminho: `<src.DIR_EXPERIMENTS>/<dataset_name>/not_incremental/<algo_name>/<timestamp_atual>`.
    '''
    not_incremental_algo_options, datasets_options = get_input(
        'Select the options for running non-incremental experiments',
        [
            {
                'name': 'algorithms',
                'description': 'Non-incremental algorithms to be executed',
                'name_column': src.NOT_INCREMENTAL_ALGORITHM_NAME,
                'options': NOT_INCREMENTAL_ALGORITHMS_TABLE
            },
            {
                'name': 'datasets',
                'description': 'Datasets to be used for training and evaluation',
                'name_column': src.DATASET_NAME,
                'options': DATASETS_TABLE
            },
        ]
    )

    for dataset_option in datasets_options:
        dataset_name = DATASETS_TABLE.loc[dataset_option, src.DATASET_NAME]
        dataset = Dataset(dataset_name)

        for not_incremental_algo_option in not_incremental_algo_options:
            NotIncrementalAlgo = NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[not_incremental_algo_option, src.NOT_INCREMENTAL_ALGORITHM_CLASS]
            not_incremental_name = NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[not_incremental_algo_option, src.NOT_INCREMENTAL_ALGORITHM_NAME]
            not_incremental_hyperparams = NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[not_incremental_algo_option, src.NOT_INCREMENTAL_ALGORITHM_HYPERPARAMETERS]

            experiment_save_path = generate_not_incremental_save_path(dataset_name, not_incremental_name)

            logger = Logger(experiment_save_path)
            logger.print_and_log(f'Running experiment for {dataset_name} with {not_incremental_name}...')

            run_experiment(algorithm=NotIncrementalAlgo, dataset=dataset, logger=logger, save_path=experiment_save_path, grid_search_params=not_incremental_hyperparams)
    print('OK!')

if __name__ == '__main__':
    main()
    sys.exit(0)
