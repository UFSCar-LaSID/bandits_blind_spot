
import os
import time

import src


def get_last_experiment_path(save_dir: str) -> str:
    return os.path.join(save_dir, str(max([int(directory) for directory in os.listdir(save_dir)])))

def generate_not_incremental_save_path(dataset_name: str, not_incremental_algo_name: str) -> str:
    return os.path.join(src.DIR_EXPERIMENTS, dataset_name, 'not_incremental', not_incremental_algo_name, str(int(time.time())))

def get_last_not_incremental_save_path(dataset_name: str, not_incremental_algo_name: str) -> str:
    return get_last_experiment_path(os.path.join(src.DIR_EXPERIMENTS, dataset_name, 'not_incremental', not_incremental_algo_name))


def generate_incremental_save_path(dataset_name: str, incremental_algo_name: str, embeddings_algo_name: str, context_name: str) -> str:
    return os.path.join(src.DIR_EXPERIMENTS, dataset_name, 'incremental', incremental_algo_name, embeddings_algo_name, context_name, str(int(time.time())))

def get_last_incremental_save_path(dataset_name: str, incremental_algo_name: str, embeddings_algo_name: str, context_name: str) -> str:
    return get_last_experiment_path(os.path.join(src.DIR_EXPERIMENTS, dataset_name, 'incremental', incremental_algo_name, embeddings_algo_name, context_name))


def generate_metrics_save_path(dataset_name: str) -> str:
    return os.path.join(src.DIR_RESULTS, dataset_name, str(int(time.time())))