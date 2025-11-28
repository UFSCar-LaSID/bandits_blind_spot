# Code adapted from: https://github.com/fidelity/mab2rec/blob/main/mab2rec/rec.py
# The difference is that the original code accept different types of arms (strings, integers, etc) and the modified code only accept sequential integers as arms. With this modification, the code can be optimized.

from typing import List, Tuple, Union
from mabwiser.utils import Arm, Num
from scipy.special import expit
from mab2rec import BanditRecommender
from mab2rec import LearningPolicy
from mabwiser.utils import Arm
import numpy as np
import pandas as pd
import torch
import src
from src.algorithms.incremental.mab2rec.lib_modifications.MABArmEncoded import MABArmEncodedOptimized

class BanditRecommenderArmEncodedOptimized(BanditRecommender):

    def __init__(self, learning_policy: Union[LearningPolicy.LinGreedy,
                                              LearningPolicy.LinTS,
                                              LearningPolicy.LinUCB],
                 num_arms: int,
                 num_features: int,
                 neighborhood_policy: Union[None] = None,
                 top_k: int = 10,
                 seed: int = src.RANDOM_STATE,
                 n_jobs: int = 1,
                 backend: str = None,
                 device: str = 'cpu'):
        """Initializes bandit recommender with the given arguments.

        Validates the arguments and raises exception in case there are violations.

        Parameters
        ----------
        learning_policy : LearningPolicy
            The learning policy.
        neighborhood_policy : NeighborhoodPolicy, default=None
            The context policy.
        top_k : int, default=10
            The number of items to recommend.
        seed : numbers.Rational, default=Constants.default_seed
            The random seed to initialize the random number generator.
            Default value is set to Constants.default_seed.value
        top_k : int, default=10
            The number of items to recommend.
        n_jobs : int, default=1
            This is used to specify how many concurrent processes/threads should be used for parallelized routines.
            If set to -1, all CPUs are used.
            If set to -2, all CPUs but one are used, and so on.
        backend : str, default=None
            Specify a parallelization backend implementation supported in the joblib library. Supported options are:
            - “loky” used by default, can induce some communication and memory overhead when exchanging input and
              output data with the worker Python processes.
            - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
            - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
              called function relies a lot on Python objects.
            Default value is None. In this case the default backend selected by joblib will be used.
        """
        super().__init__(learning_policy, neighborhood_policy, top_k, seed, n_jobs, backend)
        self.device = device
        self.num_arms = num_arms
        self.num_features = num_features
    
    def _init(self) -> None:
        """Initializes recommender with given list of arms.

        Parameters
        ----------
        arms : List[Union[Arm]]
            The list of all of the arms available for decisions.
            Arms can be integers, strings, etc.

        Returns
        -------
        Returns nothing
        """
        self.mab = MABArmEncodedOptimized(self.num_arms, self.num_features, self.learning_policy, self.neighborhood_policy, self.seed, self.n_jobs, self.backend, self.device)
    
    def fit(self, decisions: Union[List[Arm], np.ndarray, pd.Series],
            rewards: Union[List[Num], np.ndarray, pd.Series],
            contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) -> None:
        """Fits the recommender the given *decisions*, their corresponding *rewards* and *contexts*, if any.
        If the recommender arms has not been initialized using the `set_arms`, the recommender arms will be set
        to the list of arms in *decisions*.

        Validates arguments and raises exceptions in case there are violations.

        This function makes the following assumptions:
            - each decision corresponds to an arm of the bandit.
            - there are no ``None``, ``Nan``, or ``Infinity`` values in the contexts.

        Parameters
        ----------
         decisions : Union[List[Arm], np.ndarray, pd.Series]
            The decisions that are made.
         rewards : Union[List[Num], np.ndarray, pd.Series]
            The rewards that are received corresponding to the decisions.
         contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame], default=None
            The context under which each decision is made.

        Returns
        -------
        Returns nothing.
        """
        if self.mab is None:
            self._init()
        self.mab.fit(decisions, rewards, contexts)
    
    def recommend(self, contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None,
                  excluded_arms: List[List[Arm]] = None, return_scores: bool = False, apply_sigmoid: bool = True) \
            -> Union[Union[List[Arm], Tuple[List[Arm], List[Num]],
                     Union[List[List[Arm]], Tuple[List[List[Arm]], List[List[Num]]]]]]:
        #self._validate_mab(is_fit=True)
        #self._validate_get_rec(contexts, excluded_arms)

        # Get predicted expectations
        num_contexts = len(contexts) if contexts is not None else 1
        if num_contexts == 1:
            expectations = np.array([self.mab.predict_expectations(contexts)])
        else:
            expectations = self.mab.predict_expectations(contexts)
        
        #if not isinstance(expectations, np.ndarray):
        #    expectations = np.array(expectations)

        if apply_sigmoid:
            expectations = expit(expectations)

        # Create an exclusion mask, where exclusion_mask[context_ind][arm_ind] denotes if the arm with the
        # index arm_ind was excluded for context with the index context_ind.
        # The value will be True if it is excluded and those arms will not be returned as part of the results.

        # Set excluded item scores to -1, so they automatically get placed lower in best results
        expectations[excluded_arms] = -1.

        # Get best `top_k` results by sorting the expectations
        #expectations = torch.tensor(expectations, device='cuda')
        topk_sorted_expectations = torch.topk(expectations, self.top_k, dim=1)
        recommendations = topk_sorted_expectations.indices.cpu().numpy()
        scores = topk_sorted_expectations.values.cpu().numpy()

        if return_scores:
            if num_contexts > 1:
                return recommendations, scores
            else:
                return recommendations[0], scores[0]
        else:
            if num_contexts > 1:
                return recommendations
            else:
                return recommendations[0]