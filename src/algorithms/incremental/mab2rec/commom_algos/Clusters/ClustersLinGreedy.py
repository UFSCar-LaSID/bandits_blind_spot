
from src.algorithms.incremental.mab2rec.Mab2recRecommender import Mab2RecRecommender
from src.algorithms.incremental.mab2rec.lib_modifications.BanditRecommenderArmEncoded import BanditRecommenderArmEncoded
from mab2rec import LearningPolicy
from mab2rec import NeighborhoodPolicy
import src
from src.scripts.utils.Logger import Logger

class ClustersLinGreedy(Mab2RecRecommender):

    def __init__(self, user_column: str=src.COLUMN_USER_ID, item_column: str=src.COLUMN_ITEM_ID, rating_column: str=src.COLUMN_RATING, logger: Logger=None, hyperparameters: dict={}):
        super().__init__(user_column, item_column, rating_column, logger)

        self.recommender = BanditRecommenderArmEncoded(
            learning_policy=LearningPolicy.LinGreedy(**hyperparameters['learning_policy']),
            neighborhood_policy=NeighborhoodPolicy.Clusters(**hyperparameters['neighborhood_policy']),
            top_k=src.TOP_N,
            seed=src.RANDOM_STATE,
            logger=self.logger
        )
