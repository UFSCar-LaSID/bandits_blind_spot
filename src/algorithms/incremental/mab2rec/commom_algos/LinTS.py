
from src.algorithms.incremental.mab2rec.Mab2recRecommender import Mab2RecRecommenderOptimized
from src.algorithms.incremental.mab2rec.lib_modifications.BanditRecommenderArmEncoded import BanditRecommenderArmEncodedOptimized
from mab2rec import LearningPolicy
import src
from src.scripts.utils.Logger import Logger

class LinTSOptimized(Mab2RecRecommenderOptimized):

    def __init__(self, num_users: int, num_items: int, num_features: int, user_column: str=src.COLUMN_USER_ID, item_column: str=src.COLUMN_ITEM_ID, rating_column: str=src.COLUMN_RATING, hyperparameters: dict={}, logger=None):
        super().__init__(num_users, num_items, num_features, user_column, item_column, rating_column)

        self.recommender = BanditRecommenderArmEncodedOptimized(
            num_arms=self.num_items,
            num_features=self.num_features,
            learning_policy=LearningPolicy.LinTS(**hyperparameters),
            top_k=src.TOP_N,
            seed=src.RANDOM_STATE,
            device=self.device
        )