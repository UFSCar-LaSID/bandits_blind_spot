
import pandas as pd
import src

from src.algorithms.incremental.mab2rec.commom_algos.Lin import LinOptimized
from src.algorithms.incremental.mab2rec.commom_algos.LinUCB import LinUCBOptimized
from src.algorithms.incremental.mab2rec.commom_algos.LinGreedy import LinGreedyOptimized
from src.algorithms.incremental.mab2rec.commom_algos.LinTS import LinTSOptimized
#from src.algorithms.incremental.mab2rec.commom_algos.Clusters.ClustersLin import ClustersLin
#from src.algorithms.incremental.mab2rec.commom_algos.Clusters.ClustersLinUCB import ClustersLinUCB
#from src.algorithms.incremental.mab2rec.commom_algos.Clusters.ClustersLinGreedy import ClustersLinGreedy
#from src.algorithms.incremental.mab2rec.commom_algos.Clusters.ClustersLinTS import ClustersLinTS
#from src.algorithms.incremental.mab2rec.custom_algos.TimeAwareLinBoltzmann import TimeAwareLinBoltzmann
from src.algorithms.incremental import hyperparameters 

# Then define your table:
INCREMENTAL_ALGORITHMS_TABLE = pd.DataFrame(
    [
        [1, 'Lin',                    LinOptimized,                   hyperparameters.LIN],
        [2, 'LinUCB',                 LinUCBOptimized,                hyperparameters.LIN_UCB_HYPERPARAMETERS],
        [3, 'LinGreedy',              LinGreedyOptimized,             hyperparameters.LIN_GREEDY_HYPERPARAMETERS],
        [4, 'LinTS',                  LinTSOptimized,                 hyperparameters.LIN_TS_HYPERPARAMETERS],
        #[5, 'ClustersLin',            ClustersLin,           hyperparameters.CLUSTERS_LIN_HYPERPARAMETERS],
        #[6, 'ClustersLinUCB',         ClustersLinUCB,        hyperparameters.CLUSTERS_LIN_UCB_HYPERPARAMETERS],
        #[7, 'ClustersLinGreedy',      ClustersLinGreedy,     hyperparameters.CLUSTERS_LIN_GREEDY_HYPERPARAMETERS],
        #[8, 'ClustersLinTS',          ClustersLinTS,         hyperparameters.CLUSTERS_LIN_TS_HYPERPARAMETERS],
        #[9, 'TimeAwareLinBoltzmann', TimeAwareLinBoltzmann, hyperparameters.TIME_AWARE_LIN_BOLTZMANN_HYPERPARAMETERS]
    ],
    columns=[
        src.INCREMENTAL_ALGORITHM_ID,
        src.INCREMENTAL_ALGORITHM_NAME,
        src.INCREMENTAL_ALGORITHM_CLASS,
        src.INCREMENTAL_ALGORITHM_HYPERPARAMETERS
    ]
).set_index(src.INCREMENTAL_ALGORITHM_ID)
