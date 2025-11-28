
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_path)

from src.scripts.utils.parameters_handle import get_input
from src.datasets import DATASETS_TABLE
from src.algorithms.not_incremental import NOT_INCREMENTAL_ALGORITHMS_TABLE
from src.algorithms.incremental.contexts_builder import CONTEXTS_BUILDER_TABLE

import src

import pandas as pd

FILTER_ID = 'id'
FILTER_VALUE = 'qnt_interactions_per_user_filter'

FILTERS_TABLE = pd.DataFrame(
    [[1, 4],
     [2, 6],
     [3, 8]],
    columns=[FILTER_ID, FILTER_VALUE]
).set_index(FILTER_ID)



def load_dataset(dataset_name: str):
    dataset_path = os.path.join(src.DIR_DATASET, dataset_name)
    interactions_path = os.path.join(dataset_path, src.FILE_INTERACTIONS)
    
    df = pd.read_csv(interactions_path, sep=src.DELIMITER, encoding=src.ENCODING, quoting=src.QUOTING, quotechar=src.QUOTECHAR)

    df = df.sort_values(by='timestamp')
    df = df.reset_index(drop=True)
    
    return df



from src.scripts.dataset_analyses.functions import analyse_dataset

def main():
    '''
    TODO: Documentar !
    '''

    datasets_options, filters_options, embeddings_options, contexts_options = get_input(
        'Choose options for dataset analyses', 
        [
            {
                'name': 'datasets',
                'description': 'Dataset names (or indexes) to analyse. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be processed.',
                'options': DATASETS_TABLE,
                'name_column': src.DATASET_NAME
            },
            {
                'name': 'qnt_test_interactions_per_user',
                'description': 'Minimum amount of interactions to consider a user in the test set. If not provided, a interactive menu will be shown. If "all" is provided, all filters will be processed.',
                'options': FILTERS_TABLE,
                'name_column': FILTER_VALUE
            },
            {
                'name': 'embeddings',
                'description': 'Embeddings to be used in the analyses. If not provided, a interactive menu will be shown. If "all" is provided, all embeddings will be processed.',
                'options': NOT_INCREMENTAL_ALGORITHMS_TABLE,
                'name_column': src.NOT_INCREMENTAL_ALGORITHM_NAME
            },
            {
                'name': 'contexts',
                'description': 'Contexts to be used in the analyses. If not provided, a interactive menu will be shown. If "all" is provided, all contexts will be processed.',
                'options': CONTEXTS_BUILDER_TABLE,
                'name_column': src.CONTEXTS_BUILDER_NAME
            }
        ]
    )

    for dataset_option in datasets_options:
        dataset_name = DATASETS_TABLE.loc[dataset_option, src.DATASET_NAME]
        print('Loading dataset {}...'.format(dataset_name))
        dataset = load_dataset(dataset_name)

        if not 'rating' in dataset.columns:
            dataset['rating'] = 1

        for filter_option in filters_options:
            filter_value = FILTERS_TABLE.loc[filter_option, FILTER_VALUE]

            for embedding_option in embeddings_options:
                Embedding_generator = NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[embedding_option, src.NOT_INCREMENTAL_ALGORITHM_CLASS]
                embeddings_name = NOT_INCREMENTAL_ALGORITHMS_TABLE.loc[embedding_option, src.NOT_INCREMENTAL_ALGORITHM_NAME]
                
                for context_option in contexts_options:
                    build_contexts = CONTEXTS_BUILDER_TABLE.loc[context_option, src.CONTEXTS_BUILDER_FUNCTION]
                    contexts_name = CONTEXTS_BUILDER_TABLE.loc[context_option, src.CONTEXTS_BUILDER_NAME]
                    
                    processed_save_path = os.path.join(src.DIR_ANALYSES, dataset_name, embeddings_name, contexts_name)
                    save_path = os.path.join(src.DIR_ANALYSES, dataset_name, embeddings_name, contexts_name, str(filter_value))
                    print('Analysing dataset {} with embeddings {} and contexts {} (filter {})...'.format(dataset_name, embeddings_name, contexts_name, filter_value))
                    embedding_generator = Embedding_generator(hyperparameters={'factors': 10, 'random_state':1, 'num_threads':1})
                    analyse_dataset(dataset, embedding_generator, build_contexts, save_path, processed_save_path, filter_value)


if __name__ == '__main__':
    main()
    sys.exit(0)