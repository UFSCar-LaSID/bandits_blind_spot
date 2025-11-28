
from typing import Type
from sklearn.model_selection import ParameterGrid
import os

import src
from src.algorithms.incremental.mab2rec .Mab2recRecommender import Mab2RecRecommenderOptimized
from src.datasets.Dataset import Dataset
from src.scripts.utils.Logger import Logger
from src.scripts.utils.EmbeddingsLoader import EmbeddingsLoader
from src.algorithms.incremental.contexts_builder.ContextsBuilder import ContextsBuilder
from src.scripts.utils.save import save_val_recommendations, TestRecommendationsSaver, save_dict_as_json
from src.metrics.ndcg import ndcg
from src.metrics.hit_rate import hit_rate
from tqdm import tqdm
import numpy as np

def run_experiment(
        Algorithm: Type[Mab2RecRecommenderOptimized], 
        dataset: Dataset, 
        logger: Logger,
        embeddings_loader: EmbeddingsLoader,
        ContextsBuilder: Type[ContextsBuilder],
        save_path: str, 
        incremental_algo_grid_search_params: dict,
        embeddings_grid_search_params: dict,
        context_builder_grid_search_params: dict
    ):
    '''
    Executa um experimento incremental.

    Com a base de dados carregada, todas combinações de hiperparâmetros são avaliadas. No caso, todas as combinações possíveis entre hiperparâmetros do algo incremental e das embeddings serão testadas. Para essa avaliação, o algoritmo é treinado na partição de TREINO e depois avaliado na partição de VALIDAÇÃO, calculando-se uma métrica (como NDCG ou HR). Para cada combinação de hiperparams, são salvas as recomendações na partição de VALIDAÇÃO.

    Com a melhor combinação de hiperparâmetros (que obteve melhor NDCG ou HR), o algoritmo é treinado e testado de forma incremental (semelhante ao que é feito no upper bound dos algoritmos não-incrementais). Primeiro é treinado com a base de treino completa e recomendações para a primeira janela de teste são feitas. Depois, é treinado utilizando a base de treino completa + primeira janela de teste e recomendações para a segunda janela de teste. Segue dessa forma até realizar recomendações para todas as janelas. Feito isso, recomendações para todas as janelas de teste serão salvas.

    Os arquivos salvos estão melhores definidos no [README.md](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/scripts/incremental/README.md)

    params:
        algorithm: Algoritmo (não-incremental) a ser utilizado.
        dataset: Dataset a ser utilizado.
        logger: Logger a ser utilizado.
        embeddings_loader: Loader de embeddings a ser utilizado.
        build_contexts: Função que gera os contextos para cada interação.
        save_path: Caminho onde os resultados dos experimentos devem ser salvos.
        incremental_algo_grid_search_params: Dicionário contendo os hiperparâmetros do algoritmo incremental a serem testados.
        embeddings_grid_search_params: Dicionário contendo os hiperparâmetros do modelo de embeddings a serem testados.
        context_builder_grid_search_params: Dicionário contendo os hiperparâmetros da geração de contexto a serem testados. 
    '''
    grid_search_results = {}

    incremental_params_grid = list(ParameterGrid(incremental_algo_grid_search_params))
    embeddings_params_grid = list(ParameterGrid(embeddings_grid_search_params))
    contexts_params_grid = list(ParameterGrid(context_builder_grid_search_params))

    train_full_df = dataset.full_train_interactions
    train_df = dataset.train_interactions
    val_df = dataset.val_interactions

    num_users = train_full_df[src.COLUMN_USER_ID].nunique()
    num_items = train_full_df[src.COLUMN_ITEM_ID].nunique()

    # A primeira etapa é treinar o algoritmo incremental com todas as combinações de hiperparams e calcular uma métrica na partição de validação.
    # O melhor conjunto de hiperparâmetros é aquele que obter o valor da métrica mais alta, sendo usado posteriormente para testes nas janelas de teste.
    i = 0
    best_hypeparameters_idx = 0
    progress_bar = tqdm(total=len(embeddings_params_grid) * len(contexts_params_grid) * len(incremental_params_grid), desc='Grid search')
    for embeddings_params in embeddings_params_grid:
        items_embeddings = embeddings_loader.load_train_items_embeddings(embeddings_params)
        users_embeddings = embeddings_loader.load_train_users_embeddings(embeddings_params)

        for contexts_params in contexts_params_grid:
            context_builder = ContextsBuilder(users_embeddings, items_embeddings, contexts_params)
            print('generating contexts')
            train_contexts = context_builder.generate_new_contexts(train_df)
            val_contexts = context_builder.generate_new_contexts(val_df)
            num_features = train_contexts[0].shape[0]
            print('end contexts generation')

            for incremental_params in incremental_params_grid:
                logger.print_and_log(f'Current incremental params: \n{incremental_params}')

                incremental_algo = Algorithm(
                    num_users=num_users,
                    num_items=num_items,
                    num_features=num_features,
                    user_column=src.COLUMN_USER_ID,
                    item_column=src.COLUMN_ITEM_ID,
                    rating_column=src.COLUMN_RATING,
                    logger=logger,
                    hyperparameters=incremental_params
                )

                for j in range(0, len(train_df), src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE):
                    batch_df = train_df.iloc[j:j+src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE]
                    batch_contexts = train_contexts[j: j+src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE]

                    incremental_algo.train(batch_df, batch_contexts)
                
                recommendations_ids = np.empty((val_df.shape[0], src.TOP_N), dtype=int)
                recommendations_scores = np.empty((val_df.shape[0], src.TOP_N), dtype=float)

                for j in range(0, len(val_df), src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE):
                    batch_df = val_df.iloc[j:j + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE]
                    batch_contexts = val_contexts[j:j + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE]

                    batch_recommendations_ids, batch_recommendations_scores = incremental_algo.recommend(batch_df[src.COLUMN_USER_ID].values, batch_contexts)
                    recommendations_ids[j:j + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE] = batch_recommendations_ids
                    recommendations_scores[j:j + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE] = batch_recommendations_scores

                recs_df = save_val_recommendations(val_df, recommendations_ids, recommendations_scores, dataset, 
                                                   save_path=os.path.join(save_path, src.DIR_VALIDATION, str(i), src.FILE_RECOMMENDATIONS))
                current_val_ndcg = hit_rate(recs_df)

                grid_search_results[i] = {
                    "params": {
                        "incremental_params": incremental_params,
                        "embeddings_params": embeddings_params,
                        "contexts_params": contexts_params
                    },
                    "score": current_val_ndcg
                }

                if current_val_ndcg > grid_search_results[best_hypeparameters_idx]["score"]:
                    best_hypeparameters_idx = i

                i += 1
                progress_bar.update(1)
    progress_bar.close()
    logger.print_and_log('Grid search finished...\nLoading full embeddings...')

    # Após descobrir os melhores hiperparâmetros, é preciso treinar com a base full de treino (treino+val) e incrementalmente testando e treinando nas janelas de teste

    test_recommendations_saver = TestRecommendationsSaver()

    best_embeddings_params = grid_search_results[best_hypeparameters_idx]["params"]["embeddings_params"]
    items_embeddings = embeddings_loader.load_train_full_items_embeddings(best_embeddings_params)
    users_embeddings = embeddings_loader.load_train_full_users_embeddings(best_embeddings_params)

    best_contexts_params = grid_search_results[best_hypeparameters_idx]["params"]["contexts_params"]
    context_builder = ContextsBuilder(users_embeddings, items_embeddings, best_contexts_params)
    train_full_contexts = context_builder.generate_new_contexts(train_full_df)
    num_features = train_full_contexts[0].shape[0]

    best_incremental_params = grid_search_results[best_hypeparameters_idx]["params"]["incremental_params"]
    incremental_algo = Algorithm(
        num_users=num_users,
        num_items=num_items,
        num_features=num_features,
        user_column=src.COLUMN_USER_ID,
        item_column=src.COLUMN_ITEM_ID,
        rating_column=src.COLUMN_RATING,
        logger=logger,
        hyperparameters=best_incremental_params
    )
    
    for j in range(0, len(train_full_df), src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE):
        batch_df = train_full_df.iloc[j:j+src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE]
        batch_contexts = train_full_contexts[j: j+src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE]
        incremental_algo.train(batch_df, batch_contexts)

    for test_window_df in tqdm(dataset.test_splits, desc='Testing best hyperparameters', total=src.TEST_WINDOWS_QNT):
        current_test_window_contexts = context_builder.generate_new_contexts(test_window_df)

        recommendations_ids = np.empty((test_window_df.shape[0], src.TOP_N), dtype=int)
        recommendations_scores = np.empty((test_window_df.shape[0], src.TOP_N), dtype=float)

        for i in range(0, len(test_window_df), src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE):
            batch_df = test_window_df.iloc[i:i + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE]
            batch_contexts = current_test_window_contexts[i:i + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE]

            batch_recommendations_ids, batch_recommendations_scores = incremental_algo.recommend(batch_df[src.COLUMN_USER_ID].values, batch_contexts)
            recommendations_ids[i:i + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE] = batch_recommendations_ids
            recommendations_scores[i:i + src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE] = batch_recommendations_scores
        
        test_recommendations_saver.add_new_test_window(
            test_window_df,
            recommendations_ids,
            recommendations_scores
        )

        for j in range(0, len(test_window_df), src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE):
            batch_df = test_window_df.iloc[j:j+src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE]
            batch_contexts = current_test_window_contexts[j: j+src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE]
            incremental_algo.partial_train(batch_df, batch_contexts)
    
    test_recs = test_recommendations_saver.save(
        save_path=os.path.join(save_path, src.FILE_FINAL_RECOMMENDATIONS),
        dataset=dataset
    )

    save_dict_as_json(
        {
            "grid_search": grid_search_results,
            "best_hiperparam_combination_id": best_hypeparameters_idx,
            "configs": {
                "incremental_algo": str(Algorithm),
                "embeddings_algo": str(embeddings_loader.Embeddings_algo),
                "contexts_algo": str(ContextsBuilder),
                "topn": src.TOP_N,
                "random_state": src.RANDOM_STATE,
                "train_size": src.TRAIN_SIZE,
                "test_size": src.TEST_SIZE,
                "test_windows_qnt": src.TEST_WINDOWS_QNT,
                "val_size": src.VAL_SIZE,
                "incremental_algo_tested_hyperparameters": incremental_algo_grid_search_params,
                "embeddings_algo_tested_hyperparameters": embeddings_grid_search_params,
                "contexts_tested_hyperparameters": context_builder_grid_search_params,
                'incremental_items_batch_size': src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE,
                'incremental_train_batch_size': src.INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE,
                'incremental_test_batch_size': src.INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE,
            }
        },
        os.path.join(save_path, src.FILE_METADATA)
    )

    logger.print_and_log('Current experiment finished successfully !')
    logger.print_and_log(f'Results save at: {save_path}')
    logger.print_and_log(f'Final HR: {hit_rate(test_recs)}')
