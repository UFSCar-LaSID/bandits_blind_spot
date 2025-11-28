import csv

# Diretorios e caminhos
DIR_RAW = 'raw'
DIR_DATASET = '1-datasets'
DIR_EXPERIMENTS = '2-experiments'
DIR_RESULTS = '3-results'
DIR_TEX = '4-tex'
DIR_IMAGES = '5-images'
DIR_ANALYSES = '6-analyses'
DIR_AGG_RESULTS = '7-agg_results'


DIR_NOT_INCREMENTAL = 'not_incremental'
DIR_INCREMENTAL = 'incremental'
DIR_TRAIN_EMBEDDINGS = 'train_embeddings'
DIR_TRAIN_FULL_EMBEDDINGS = 'train_full_embeddings'
DIR_VALIDATION = 'validation'

# Nomes de arquivos internos
FILE_SETUP = 'setup.json'

# Nomes de arquivos
FILE_INTERACTIONS = 'interactions.csv'
FILE_ITEMS = 'items.csv'
FILE_USERS = 'users.csv'
FILE_METADATA = 'metadata.json'
FILE_LOGS = 'logs.txt'
FILE_RECOMMENDATIONS = 'recommendations.csv'
FILE_RECOMMENDATIONS_LOWER = 'recommendations_lower.csv'
FILE_RECOMMENDATIONS_UPPER = 'recommendations_upper.csv'
FILE_FINAL_RECOMMENDATIONS = 'final_recommendations.csv'
FILE_USERS_EMBEDDINGS = 'users_embeddings.npy'
FILE_ITEMS_EMBEDDINGS = 'items_embeddings.npy'
FILE_FULL_TEST_METRICS = 'full_test_metrics.csv'
FILE_WINDOW_MEAN_METRICS = 'window_mean_metrics.csv'
FILE_PER_WINDOW_GRAPHIC_HTML = 'per_window_graphic.html'
FILE_PER_WINDOW_GRAPHIC_PNG = 'per_window_graphic.png'
FILE_PER_WINDOW_TABLE = 'per_window_table.csv'
FILE_WINDOW_AGG_GRAPHIC_HTML = 'window_agg_graphic.html'
FILE_WINDOW_AGG_GRAPHIC_PNG = 'window_agg_graphic.png'
FILE_WINDOW_AGG_TABLE = 'window_agg_table.csv'

# Colunas dos CSV e DataFrames
COLUMN_ITEM_ID = 'id_item'
COLUMN_USER_ID = 'id_user'
COLUMN_RATING = 'rating'
COLUMN_IS_POSITIVE = 'is_positive'
COLUMN_DATETIME = 'datetime'
COLUMN_TIMESTAMP = 'timestamp'
COLUMN_RECOMMENDATIONS = 'recommendations'
COLUMN_SCORES = 'scores'
COLUMN_WINDOW_NUMBER = 'window_number'
COLUMN_TIMEDIFF = 'timediff'

# Dados dos CSVs
DELIMITER = ';'
ENCODING = "utf-8"
QUOTING = csv.QUOTE_ALL
QUOTECHAR = '"'

# Colunas da tabela de algoritmos não incrementais
NOT_INCREMENTAL_ALGORITHM_ID = 'id'
NOT_INCREMENTAL_ALGORITHM_NAME = 'name'
NOT_INCREMENTAL_ALGORITHM_CLASS = 'class'
NOT_INCREMENTAL_ALGORITHM_HYPERPARAMETERS = 'hyperparameters'

# Colunas da tabela de algoritmos incrementais
INCREMENTAL_ALGORITHM_ID = 'id'
INCREMENTAL_ALGORITHM_NAME = 'name'
INCREMENTAL_ALGORITHM_CLASS = 'class'
INCREMENTAL_ALGORITHM_HYPERPARAMETERS = 'hyperparameters'

# Colunas da tabela de construtores de contextos
CONTEXTS_BUILDER_ID = 'id'
CONTEXTS_BUILDER_NAME = 'name'
CONTEXTS_BUILDER_CLASS = 'function'
CONTEXTS_BUILDER_HYPERPARAMETERS = 'hyperparameters'

# Colunas da tabela de datasets
DATASET_ID = 'id'
DATASET_NAME = 'name'
DATASET_TYPE = 'type'
DATASET_MISSING_RATINGS_FUNC = 'missing_ratings_func'
DATASET_IS_POSITIVE_FUNC = 'is_positive_func'
DATASET_PREPROCESS_FUNCTION = 'function'

# Colunas da tabela de métricas
METRIC_ID = 'id'
METRIC_NAME = 'name'
METRIC_FUNCTION = 'function'

# Parametros gerais dos experimentos
TOP_N = 20
RANDOM_STATE = 42
TEST_SIZE = 0.5
TRAIN_SIZE = 1 -TEST_SIZE
TEST_WINDOWS_QNT = 10
VAL_SIZE = 'same size as test window'

GRAPHIC_COLORS = [
    'brown',
    'blue',
    'black',
    'red',
    'green',
    'purple',
    'orange',
    'pink',
    'blue',
    'red'
]
EXECUTION_MODE = 'cpu' # 'cpu' or 'gpu'


# Incremental configs
INCREMENTAL_ALGO_ITEMS_BATCH_SIZE = 500
INCREMENTAL_ALGO_TRAIN_INTERACTIONS_BATCH_SIZE = 500
INCREMENTAL_ALGO_TEST_INTERACTIONS_BATCH_SIZE = 500

# Config do experimento
EXPERIMENT_CONFIG = {
    'top_n': TOP_N,
    'random_state': RANDOM_STATE,
    'test_size': TEST_SIZE,
    'train_size': TRAIN_SIZE,
    'test_windows_qnt': TEST_WINDOWS_QNT,
    'val_size': VAL_SIZE,
    'execution_mode': EXECUTION_MODE
}