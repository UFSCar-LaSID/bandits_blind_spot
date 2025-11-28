
import os
from typing import TypedDict, Dict, Callable
from collections import defaultdict
import itertools
import ast

import pandas as pd
import numpy as np

import src
from src.scripts.utils.get_save_path import get_last_not_incremental_save_path, get_last_incremental_save_path
from src.scripts.utils.EmbeddingsLoader import EmbeddingsLoader
from src.datasets.Dataset import Dataset



class WindowSpecificMetricDict(TypedDict):
    per_window: pd.DataFrame
    window_agg: pd.DataFrame

class MetricsDict(TypedDict):
    metrics: 'dict[str, WindowSpecificMetricDict]'
    window_mean: pd.DataFrame       
    test_full: pd.DataFrame


def calculate_all_metrics(
    dataset: Dataset,
    dataset_name: str,
    not_incremental_algo_names: 'list[str]',
    not_incremental_algos,
    incremental_algo_names: 'list[str]',
    embeddings_generators_names: 'list[str]',
    embeddings_generators,
    context_generators_names: 'list[str]',
    metrics: 'dict[str, Callable[[pd.DataFrame], float]]'
) -> MetricsDict:
    '''
    Calcula todas as métricas necessárias para gerar as tabelas e gráficos.

    params:
        dataset_name: Nome do dataset
        not_incremental_algo_names: Lista com os nomes dos algoritmos não incrementais
        incremental_algo_names: Lista com os nomes dos algoritmos incrementais
        embeddings_generators_names: Lista com os nomes dos geradores de embeddings
        context_generators_names: Lista com os nomes dos geradores de contexto
        metrics: Dicionário com as métricas a serem calculadas. A chave é o nome da métrica e o valor é a função que calcula a métrica

    returns:
        Um dicionário seguindo a seguinte estrutura:

        ```
        {
            "metrics": {
                "<metric_name>": {
                    "per_window": "pd.DataFrame",
                    "window_agg": "pd.DataFrame"
                }
            },
            "window_mean": "pd.DataFrame",
            "test_full": "pd.DataFrame"
        } 
        ```

        Os DataFrames retornados são descritos a seguir:
        
        - `per_window`: é calculado a métrica para cada janela de teste (um DataFrame para cada métrica). As colunas são as janelas, as linhas os algoritmos

        - `window_agg`: cada janela é o resultado médio das janelas anteriores até a janela atual. Por exemplo, o resultado da janela 3 é o resultado médio das janelas 1, 2 e 3. Um DataFrame é criado para cada métrica. As colunas são as janelas, as linhas os algoritmos

        - `window_mean`: é uma tabela que possui a média dos valores obtidos em todas as janelas. As linhas são os algoritmos e as colunas são as diferentes métricas (apenas um DataFrame para todas as métricas)

        - `test_full`: são calculadas as métricas para todo o conjunto de teste (todas as janelas de testes "concatenadas"). As colunas são as métricas, as linhas os algoritmos (apenas um DataFrame para todas as métricas)
    '''

    def read_dataframe(load_path: str) -> pd.DataFrame:
        df = pd.read_csv(load_path, delimiter=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR, 
                         converters={src.COLUMN_RECOMMENDATIONS: ast.literal_eval, src.COLUMN_SCORES: ast.literal_eval})
        return df

    windows_numbers = [i for i in range(src.TEST_WINDOWS_QNT)]
    windows_cols = ['algo_name'] + windows_numbers
    results_dict = {}
    results_dict['metrics'] = {}
    embeddings_generators_infos = [(embedding_generator_name, embedding_generator) for embedding_generator_name, embedding_generator in zip(embeddings_generators_names, embeddings_generators)]

    window_mean_dict = defaultdict(dict)
    full_test_dict = defaultdict(dict)

    incremental_algo_iter = list(itertools.product(incremental_algo_names, embeddings_generators_infos, context_generators_names))

    for metric_name, calculate_metric in metrics.items():
        print(f'Calculating metric {metric_name}...')
        per_window_dict = {window_col: [] for window_col in windows_cols}
        window_agg_dict = {window_col: [] for window_col in windows_cols}

        # not_incremental
        for not_incremental_algo_name, not_incremental_algo in zip(not_incremental_algo_names, not_incremental_algos):
            print(f'Calculating not incremental algo {not_incremental_algo_name}...')
            not_incremental_save_path = get_last_not_incremental_save_path(dataset_name, not_incremental_algo_name)
            embeddings_loader = EmbeddingsLoader(not_incremental_algo, not_incremental_save_path, dataset)
            items_embeddings = embeddings_loader.load_train_full_items_embeddings(embeddings_loader.get_best_hyperparameters())

            # lower
            lower_algo_name = f'{not_incremental_algo_name} - lower'
            per_window_dict['algo_name'].append(lower_algo_name)
            window_agg_dict['algo_name'].append(lower_algo_name)
            
            not_incremental_save_path = get_last_not_incremental_save_path(dataset_name, not_incremental_algo_name)

            recs_df = read_dataframe(os.path.join(not_incremental_save_path, src.FILE_RECOMMENDATIONS_LOWER))

            for window_num in windows_numbers:
                per_window_dict[window_num].append(
                    calculate_metric(
                        recs_df[recs_df[src.COLUMN_WINDOW_NUMBER] == window_num],
                        dataset=dataset,
                        items_embeddings=items_embeddings
                    )
                )
                window_agg_dict[window_num].append(
                    np.mean([per_window_dict[i][-1] for i in range(window_num+1)])
                )
            
            full_test_dict[lower_algo_name][metric_name] = calculate_metric(
                recs_df,
                dataset=dataset,
                items_embeddings=items_embeddings
            )
            window_mean_dict[lower_algo_name][metric_name] = window_agg_dict[windows_numbers[-1]][-1]

            # upper
            upper_algo_name = f'{not_incremental_algo_name} - upper'
            per_window_dict['algo_name'].append(upper_algo_name)
            window_agg_dict['algo_name'].append(upper_algo_name)
            
            not_incremental_save_path = get_last_not_incremental_save_path(dataset_name, not_incremental_algo_name)

            recs_df = read_dataframe(os.path.join(not_incremental_save_path, src.FILE_RECOMMENDATIONS_UPPER))

            for window_num in windows_numbers:
                per_window_dict[window_num].append(
                    calculate_metric(
                        recs_df[recs_df[src.COLUMN_WINDOW_NUMBER] == window_num],
                        dataset=dataset,
                        items_embeddings=items_embeddings
                    )
                )
                window_agg_dict[window_num].append(
                    np.mean([per_window_dict[i][-1] for i in range(window_num+1)])
                )
                        
            full_test_dict[upper_algo_name][metric_name] = calculate_metric(
                recs_df,
                dataset=dataset,
                items_embeddings=items_embeddings
            )
            window_mean_dict[upper_algo_name][metric_name] = window_agg_dict[windows_numbers[-1]][-1]
        

        # incremental
        for incremental_algo_name, embeddings_generator_info, context_name in incremental_algo_iter:
            print(f'Calculating incremental algo {incremental_algo_name}...')
            embeddings_name = embeddings_generator_info[0]
            embeddings_generator = embeddings_generator_info[1]

            incremental_name = incremental_algo_name
            if len(embeddings_generators_names) > 1:
                incremental_name += f' - {embeddings_name}'
            if len(context_generators_names) > 1:
                incremental_name += f' - {context_name}'
            
            per_window_dict['algo_name'].append(incremental_name)
            window_agg_dict['algo_name'].append(incremental_name)
            
            incremental_save_path = get_last_incremental_save_path(dataset_name, incremental_algo_name, embeddings_name, context_name)
            embeddings_save_path = get_last_not_incremental_save_path(dataset_name, embeddings_name)
            embeddings_loader = EmbeddingsLoader(embeddings_generator, embeddings_save_path, dataset)
            items_embeddings = embeddings_loader.load_train_full_items_embeddings(embeddings_loader.get_best_hyperparameters())

            recs_df = read_dataframe(os.path.join(incremental_save_path, src.FILE_FINAL_RECOMMENDATIONS))

            for window_num in windows_numbers:
                per_window_dict[window_num].append(
                    calculate_metric(
                        recs_df[recs_df[src.COLUMN_WINDOW_NUMBER] == window_num],
                        dataset=dataset,
                        items_embeddings=items_embeddings
                    )
                )
                window_agg_dict[window_num].append(
                    np.mean([per_window_dict[i][-1] for i in range(window_num+1)])
                )
            
            full_test_dict[incremental_name][metric_name] = calculate_metric(
                recs_df,
                dataset=dataset,
                items_embeddings=items_embeddings
            )
            window_mean_dict[incremental_name][metric_name] = window_agg_dict[windows_numbers[-1]][-1]


        algos_names = per_window_dict['algo_name']
        per_window_dict_separated_windows_dict = {'algo_name' : [], 'window_number': [], 'value': []}
        for window_num in windows_numbers:
            per_window_dict_separated_windows_dict['algo_name'] += algos_names
            per_window_dict_separated_windows_dict['window_number'] += [window_num for _ in range(len(algos_names))]
            per_window_dict_separated_windows_dict['value'] += per_window_dict[window_num]
        
        algos_names = window_agg_dict['algo_name']
        window_agg_dict_separated_windows_dict = {'algo_name' : [], 'window_number': [], 'value': []}
        for window_num in windows_numbers:
            window_agg_dict_separated_windows_dict['algo_name'] += algos_names
            window_agg_dict_separated_windows_dict['window_number'] += [window_num for _ in range(len(algos_names))]
            window_agg_dict_separated_windows_dict['value'] += window_agg_dict[window_num]

        cur_metric_dict = {}
        cur_metric_dict['per_window'] = pd.DataFrame(per_window_dict_separated_windows_dict)
        cur_metric_dict['window_agg'] = pd.DataFrame(window_agg_dict_separated_windows_dict)
        results_dict['metrics'][metric_name] = cur_metric_dict

    
    window_mean_df = pd.DataFrame(window_mean_dict)
    window_mean_df = window_mean_df.reset_index(names='algo_name')

    full_test_df = pd.DataFrame(full_test_dict)
    full_test_df = full_test_df.reset_index(names='algo_name')
    
    results_dict['window_mean'] = window_mean_df
    results_dict['test_full'] = full_test_df

    return results_dict