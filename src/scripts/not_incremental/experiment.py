import numpy as np
import os
import pandas as pd
from sklearn.model_selection import ParameterGrid
import time
from tqdm import tqdm

import src
from src.algorithms.not_incremental.implicit.ImplicitRecommender import ImplicitRecommender
from src.datasets import Dataset
from src.scripts.utils import Logger
from src.scripts.utils.Logger import format_elapsed_time
from src.scripts.utils.save import save_dict_as_json, save_val_recommendations, TestRecommendationsSaver
from src.metrics.ndcg import batch_ndcg


def run_experiment(algorithm: ImplicitRecommender, dataset: Dataset, logger: Logger, save_path: str, grid_search_params: dict):
    '''
    Executa um experimento não incremental.

    O experimento consiste em fazer uma busca em grade nos hiperparâmetros fornecidos. Para cada combinação de hiperparâmetros, faria o seguinte:

    - **lower_bound:** Treinaria o algoritmo **APENAS com o treino**, geraria as recomendações para a partição de validação. A melhor combinação de hiperparâmetros será usada para gerar as recomendações para todas as partições de testes. As embeddings serão salvas para todas as combinações de hiperparâmetros, utilizando a partição de treino completa
    - **upper_bound:** Apenas com a melhor combinação de hiperparâmetros, treinaria o algoritmo **COM TODAS as janelas, iterativamente**, geraria as recomendações para **todas as partições de teste. NÃO irá guardar nenhuma embedding aqui.**

    Os arquivos salvos estão melhores definidos no [README.md](https://github.com/RecSys-UFSCar/mab-recsys/blob/main/src/scripts/not_incremental/README.md)

    params:
        algorithm: Algoritmo (não-incremental) a ser utilizado.
        dataset: Dataset a ser utilizado.
        logger: Logger a ser utilizado.
        save_path: Caminho onde os resultados dos experimentos devem ser salvos.
        grid_search_params: Dicionário contendo os hiperparâmetros a serem testados. É um dicionário onde a chave é o nome do hiperparâmetro e o valor é uma lista com os valores a serem testados. Todas possíveis combinações de hiperparâmetros serão testadas.
    '''
    # Metadata dict
    metadata = {
        'grid_search': dict(),
        'best_hiperparam_combination_id': None,
        'configs': src.EXPERIMENT_CONFIG
    }

    # =========================================== VALIDATION ===========================================
    # Define save path
    validation_save_path = os.path.join(save_path, src.DIR_VALIDATION)
    os.makedirs(validation_save_path, exist_ok=True)

    # Run grid search and find best parameters
    best_ndcg = -np.inf
    best_hyperparameters = None
    best_hiperparam_combination_id = None
    logger.print_and_log('Running grid search...')
    start = time.time()
    param_grid = list(ParameterGrid(grid_search_params))
    
    # Iterate over all hyperparameters combinations
    for hiperparam_combination_id, hyperparameters in tqdm(enumerate(param_grid), total=len(param_grid)):
        # Train model
        model = algorithm(hyperparameters=hyperparameters)
        model.train(dataset.train_interactions, dataset.num_users, dataset.num_items, verbose=False)

        # Filter negative interactions and generate recommendations
        recommendations_ids, recommendations_scores = model.recommend(users_ids=dataset.val_interactions[src.COLUMN_USER_ID].values, topn=src.TOP_N)
        
        # Calculate metric and compare with best
        predictions_df = dataset.val_interactions.copy()
        predictions_df[src.COLUMN_RECOMMENDATIONS] = list(recommendations_ids)
        ndcg_score = batch_ndcg(predictions_df)
        metadata['grid_search'][hiperparam_combination_id] = {'params': hyperparameters, 'score': ndcg_score}
        if ndcg_score > best_ndcg:
            best_ndcg = ndcg_score
            best_hyperparameters = hyperparameters
            best_hiperparam_combination_id = hiperparam_combination_id
        
        # Save recommendations
        hiperparam_save_path = os.path.join(validation_save_path, str(hiperparam_combination_id))
        os.makedirs(hiperparam_save_path, exist_ok=True)
        save_val_recommendations(
            val_interactions_df=dataset.val_interactions,
            recommendations_ids=recommendations_ids,
            recommendations_scores=recommendations_scores,
            dataset=dataset,
            save_path=os.path.join(hiperparam_save_path, src.FILE_RECOMMENDATIONS)
        )
        
        # Save embeddings
        embeddings_save_path = os.path.join(hiperparam_save_path, src.DIR_TRAIN_EMBEDDINGS)
        os.makedirs(embeddings_save_path, exist_ok=True)
        model.save_embeddings(embeddings_save_path)
    
    # Save best results
    end = time.time()
    logger.print_and_log(f'Grid search finished in {format_elapsed_time(start, end)}.')
    logger.print_and_log(f'Best hyperparameters: {best_hyperparameters}')
    logger.print_and_log(f'Best score: {best_ndcg}')
    metadata['best_hiperparam_combination_id'] = best_hiperparam_combination_id
    save_dict_as_json(metadata, os.path.join(save_path, src.FILE_METADATA))
    # ============================================================================================

    # =========================================== TEST ===========================================
    # Train lower bound model
    logger.print_and_log('Training lower bound model...')
    start = time.time()
    lower_bound_model = algorithm(hyperparameters=best_hyperparameters)
    lower_bound_model.train(dataset.full_train_interactions, dataset.num_users, dataset.num_items, verbose=True)

    # Save embeddings
    full_embeddings_save_path = os.path.join(validation_save_path, str(best_hiperparam_combination_id), src.DIR_TRAIN_FULL_EMBEDDINGS)
    os.makedirs(full_embeddings_save_path, exist_ok=True)
    lower_bound_model.save_embeddings(save_path=full_embeddings_save_path)
    end = time.time()
    logger.print_and_log(f'Lower bound model trained in {format_elapsed_time(start, end)}.')

    # Copy lower bound to upper bound for first iteration
    upper_bound_model = lower_bound_model
    upper_bound_interactions = dataset.full_train_interactions.copy()
    
    # Generate recommendations for test partitions
    logger.print_and_log('Generating recommendations for test splits...')
    start = time.time()
    test_saver_lower = TestRecommendationsSaver()
    test_saver_upper = TestRecommendationsSaver()
    for test_interactions in tqdm(dataset.test_splits, total=src.TEST_WINDOWS_QNT):
        # Lower bound recommendation
        recommendations_ids, recommendations_scores = lower_bound_model.recommend(users_ids=test_interactions[src.COLUMN_USER_ID].values, topn=src.TOP_N)
        test_saver_lower.add_new_test_window(
            test_window_interactions_df=test_interactions, 
            recommendations_ids=recommendations_ids, 
            recommendations_scores=recommendations_scores
        )
        # Upper bound recommendation
        recommendations_ids, recommendations_scores = upper_bound_model.recommend(users_ids=test_interactions[src.COLUMN_USER_ID].values, topn=src.TOP_N)
        test_saver_upper.add_new_test_window(
            test_window_interactions_df=test_interactions, 
            recommendations_ids=recommendations_ids, 
            recommendations_scores=recommendations_scores
        )
        # Update upper bound model
        upper_bound_interactions = pd.concat([upper_bound_interactions, test_interactions], axis=0)
        upper_bound_model = algorithm(hyperparameters=best_hyperparameters)
        upper_bound_model.train(upper_bound_interactions, dataset.num_users, dataset.num_items, verbose=False)

    # Save test recommendations
    test_saver_lower.save(save_path=os.path.join(save_path, src.FILE_RECOMMENDATIONS_LOWER), dataset=dataset)
    test_saver_upper.save(save_path=os.path.join(save_path, src.FILE_RECOMMENDATIONS_UPPER), dataset=dataset)
    end = time.time()
    logger.print_and_log(f'Test recommendations generated in {format_elapsed_time(start, end)}.')
    # ============================================================================================