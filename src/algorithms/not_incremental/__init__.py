
import pandas as pd
import src

from src.algorithms.not_incremental import hyperparameters
from src.algorithms.not_incremental.implicit.als import AlternatingLeastSquares
from src.algorithms.not_incremental.implicit.bpr import BayesianPersonalizedRanking


NOT_INCREMENTAL_ALGORITHMS_TABLE = pd.DataFrame(
    [[1, 'als',         AlternatingLeastSquares,          hyperparameters.ALS_HYPERPARAMETERS],
     [2, 'bpr',         BayesianPersonalizedRanking,      hyperparameters.BPR_HYPERPARAMETERS]],
    columns=[src.NOT_INCREMENTAL_ALGORITHM_ID, src.NOT_INCREMENTAL_ALGORITHM_NAME, src.NOT_INCREMENTAL_ALGORITHM_CLASS, src.NOT_INCREMENTAL_ALGORITHM_HYPERPARAMETERS]
).set_index(src.NOT_INCREMENTAL_ALGORITHM_ID)
