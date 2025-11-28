import src

import pandas as pd

from src.metrics.ndcg import ndcg
from src.metrics.hit_rate import hit_rate
from src.metrics.f1_score import f1_score
from src.metrics.novelty import novelty
from src.metrics.coverage import coverage
from src.metrics.diversity import diversity


METRICS_TABLE = pd.DataFrame(
    [[1,    'ncdg',              ndcg],
     [2,    'hit rate (hr)',     hit_rate],
     [3,    'f-score',           f1_score],
     [4,    'novelty',           novelty],
     [5,    'coverage',          coverage],
     [6,    'diversity',         diversity]],
    columns=[src.METRIC_ID, src.METRIC_NAME, src.METRIC_FUNCTION]
).set_index(src.METRIC_ID)
