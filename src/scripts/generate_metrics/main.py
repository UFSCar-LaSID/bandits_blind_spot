
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_path)

import src
from src.scripts.utils.parameters_handle import get_input
from src.algorithms.incremental import INCREMENTAL_ALGORITHMS_TABLE
from src.datasets import DATASETS_TABLE
from src.algorithms.not_incremental import NOT_INCREMENTAL_ALGORITHMS_TABLE
from src.algorithms.incremental.contexts_builder import CONTEXTS_BUILDER_TABLE
from src.metrics import METRICS_TABLE
from src.scripts.generate_metrics.metrics.calculate_all_metrics import calculate_all_metrics
from src.scripts.generate_metrics.graphics.generate_per_window_graphic import generate_per_window_graphic
from src.scripts.generate_metrics.graphics.generate_window_agg_graphic import generate_window_agg_graphic
from src.scripts.utils.get_save_path import generate_metrics_save_path
from src.scripts.utils.save import save_dataframe, save_dict_as_json
from src.datasets.Dataset import Dataset


def main():
    '''
    Responsável por gerar as métricas para avaliação dos algoritmos. Serão geradas tabelas e gráficos com métricas entre diferentes algoritmos em diferentes bases de dados. 

    Este código deve pedir os seguintes dados:

    1. Base de dados (apenas uma)
    2. Métricas (pode escolher mais de um)
    3. Escolher os algoritmos não-incremental (pode escolher mais de um, e até mesmo não escolher nenhum)
    4. Escolher os algoritmos incrementais (pode escolher mais de um, e até mesmo não escolher nenhum)
    5. Escolher o gerador de embeddings dos algoritmos incrementais (pode escolher mais de um)
    6. Escolher o construtor de contextos dos algoritmos incrementais (pode escolher mais de um)

    O input pode ser coletado via linha de comando (python `input()`) ou passado como argumento na execução do script (`--setup <setup_file>`).
    O arquivo de setup deve ser um arquivo JSON com a seguinte estrutura:

    ```
    {
        "dataset": 1,                           // Dataset a ser avaliado (IDs de datasets - disponíveis em src/datasets/__init__.py)
        "metrics": [1, 2],                      // Lista de métricas a serem avaliadas (IDs de métricas - disponíveis em src/metrics/__init__.py)
        "not_incremental_algorithms": [2, 3],   // Lista de algoritmos não-incrementais a serem avaliados (IDs de algoritmos - disponíveis em src/not_incremental/embeddings/__init__.py)
        "incremental_algorithms": [1, 2],       // Lista de algoritmos de incrementais a serem avaliados (IDs de algoritmos - disponíveis em src/algorithms/incremental/__init__.py)
        "embeddings": [1, 2],                   // Lista de algoritmos de embeddings a serem avaliados (IDs de algoritmos - disponíveis em src/not_incremental/embeddings/__init__.py)
        "contexts": [1, 2],                     // Lista de funções de geração de contextos a serem avaliadas (IDs de funções - disponíveis em src/algorithms/incremental/contexts_builder/__init__.py)
    }
    ```

    Os nomes dos algoritmos não-incrementais serão os mesmos que estão cadastrados na tabela de opções. Estes serão acrescentados de `upper` se estes foram treinados usando as janelas anteriores de treino. Caso contrário, se usaram apenas a partição de treino, terá o pós-fixo de `lower` (nome final: `<algo-nao-incremental-name> - <upper ou lower>`)

    O nome dos algoritmos incrementais que apareceram dependerão das opções de geração de embeddings escolhidas e de construtor de contexto. O nome dos algoritmos incrementais segue o seguinte formato: `<algo-incremental-name> - <gerador-embeddings?> - <construtor-de-embeddings?>` (se for escolhido apenas um gerador de embeddings ou construtor de embeddings, o nome destes serão removidos do nome final do algoritmo incremental).

    Após a seleção das opções, as seguintes informações serão geradas:

    - `per_window_graphic` (html e png): para cada janela é calculado a métrica para todos os algoritmos possíveis. O gráfico terá linhas com o desempenho dos algoritmos ao longo das janelas. OBS: Um gráfico desse é gerado para cada tipo de métrica selecionado.
    - `window_agg_graphic` (html e png): cada janela é o resultado médio das janelas anteriores até a janela atual. Por exemplo, o resultado da janela 3 é o resultado médio das janelas 1, 2 e 3. OBS: Um gráfico desse é gerado para cada tipo de métrica selecionado.
    - `per_window_table` (csv): basicamente, possui as mesmas informações usadas para gerar o gráfico `per_window_graphic`. As linhas são cada algoritmo e as colunas são os resultados para cada janela. OBS: Também é gerado para cada métrica selecionada
    - `window_agg_table` (csv): basicamente, possui as mesmas informações usadas para gerar o gráfico `window_agg_graphic`. As linhas são cada algoritmo e as colunas são os resultados para cada janela agregada. OBS: Também é gerado para cada métrica selecionada.
    - `window_mean_metrics` (csv): é uma tabela que possui a média dos valores obtidos em todas as janelas. As linhas são os algoritmos e as colunas são as diferentes métricas escolhidas.
    - `full_test_metrics` (csv): é uma tabela que possui a métrica calculada para a base completa de teste (não faz a média das janelas, isso pode mudar um pouco o resultado). As linhas são os algoritmos e as colunas são as diferentes métricas escolhidas.

    OBS: é necessário que os algoritmos já tenham sido executados e gerado as recomendações. Caso contrário, um erro deve acontecer.

    Os resultados serão salvos no caminho: `<src.RESULTS_DIR>/<dataset_name>/<timestamp_atual>`.
    '''
    datasets_options, metrics_options, not_incremental_algos_options, incremental_algos_options, embeddings_options, contexts_options = get_input(
        'Selecione as opções para rodar experimentos incrementais (MAB)',
        [
            {
                'name': 'datasets',
                'description': 'Datasets que serão usados para gerar as métricas. As métricas serão geradas individualmente para cada base de dados escolhida',
                'name_column': src.DATASET_NAME,
                'options': DATASETS_TABLE
            },
            {
                'name': 'metrics',
                'description': 'Métricas que serão usadas na avaliação',
                'name_column': src.METRIC_NAME,
                'options': METRICS_TABLE
            },
            {
                'name': 'not_incremental algorithms',
                'description': 'Algoritmos não-incrementais que serão usados para a comparação',
                'name_column': src.NOT_INCREMENTAL_ALGORITHM_NAME,
                'options': NOT_INCREMENTAL_ALGORITHMS_TABLE
            },
            {
                'name': 'incremental algorithms',
                'description': 'Algoritmos incrementais que serão usados para a comparação',
                'name_column': src.INCREMENTAL_ALGORITHM_NAME,
                'options': INCREMENTAL_ALGORITHMS_TABLE
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

    datasets_names = [DATASETS_TABLE.loc[dataset_option, src.DATASET_NAME] for dataset_option in datasets_options]
    metrics_dict = {METRICS_TABLE.loc[metric_option, src.METRIC_FUNCTION].__name__: METRICS_TABLE.loc[metric_option, src.METRIC_FUNCTION] for metric_option in metrics_options}
    not_incremental_algo_names = [NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[not_incremental_option, src.NOT_INCREMENTAL_ALGORITHM_NAME] for not_incremental_option in not_incremental_algos_options]
    not_incremental_algos = [NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[not_incremental_option, src.NOT_INCREMENTAL_ALGORITHM_CLASS] for not_incremental_option in not_incremental_algos_options]
    incremental_algo_names = [INCREMENTAL_ALGORITHMS_TABLE.loc[incremental_option, src.INCREMENTAL_ALGORITHM_NAME] for incremental_option in incremental_algos_options]
    embeddings_names = [NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[embeddings_option, src.NOT_INCREMENTAL_ALGORITHM_NAME] for embeddings_option in embeddings_options]
    embeddings_algos = [NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[embeddings_option, src.NOT_INCREMENTAL_ALGORITHM_CLASS] for embeddings_option in embeddings_options]
    contexts_names = [CONTEXTS_BUILDER_TABLE.loc[context_option, src.CONTEXTS_BUILDER_NAME] for context_option in contexts_options]

    for dataset_name in datasets_names:
        print(f'Loading {dataset_name} ...')
        dataset = Dataset(dataset_name, load_timediff=False)
        print(f'Dataset {dataset_name} loaded')
        print('Calculating metrics...')
        metricst_dict = calculate_all_metrics(
            dataset=dataset,
            dataset_name=dataset_name, 
            not_incremental_algo_names=not_incremental_algo_names,
            not_incremental_algos=not_incremental_algos,
            incremental_algo_names=incremental_algo_names, 
            embeddings_generators_names=embeddings_names,
            embeddings_generators=embeddings_algos,
            context_generators_names=contexts_names, 
            metrics=metrics_dict
        )

        results_save_path = generate_metrics_save_path(dataset_name)
        os.makedirs(results_save_path)

        print('Saving results...')
        generate_per_window_graphic(
            save_path=results_save_path,
            calcultated_metrics=metricst_dict
        )
        generate_window_agg_graphic(
            save_path=results_save_path,
            calcultated_metrics=metricst_dict
        )

        save_dataframe(
            df=metricst_dict['test_full'],
            save_path=os.path.join(results_save_path, src.FILE_FULL_TEST_METRICS)
        )
        save_dataframe(
            df=metricst_dict['window_mean'],
            save_path=os.path.join(results_save_path, src.FILE_WINDOW_MEAN_METRICS)
        )

        save_dict_as_json(
            {
                "metrics_names": list(metrics_dict.keys()),
                "not_incremental_algo_names": not_incremental_algo_names,
                "incremental_algos": {
                    "algo_names": incremental_algo_names,
                    "embeddings_names": embeddings_names,
                    "contexts_builders_names": contexts_names
                },
                "configs": {
                    "topn": src.TOP_N,
                    "random_state": src.RANDOM_STATE,
                    "train_size": src.TRAIN_SIZE,
                    "test_size": src.TEST_SIZE,
                    "test_windows_qnt": src.TEST_WINDOWS_QNT,
                    "val_size": src.VAL_SIZE
                }
            },
            os.path.join(results_save_path, src.FILE_METADATA)
        )
        print('Done.')



if __name__ == '__main__':
    main()
    sys.exit(0)