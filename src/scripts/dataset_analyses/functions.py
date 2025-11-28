from mab2rec import BanditRecommender, LearningPolicy
import src

import pandas as pd
import numpy as np
from tqdm import tqdm

from mabwiser.utils import create_rng
import mab2rec.utils

from mabwiser.linear import _Linear
from mabwiser.utils import Num, _BaseRNG
from typing import List, Optional

import matplotlib.pyplot as plt


def get_concat_context(interactions, context_cols):
    # Concat multiple array columns into a single array column
    return np.array(interactions[context_cols].apply(lambda x: np.concatenate(x), axis=1).tolist())

def partial_fit_mab(mab_algo, df_extra_train_with_contexts, contexts_col):
    contexts = get_concat_context(df_extra_train_with_contexts, contexts_col)
    mab_algo.mab._imp.rng = create_rng(mab2rec.utils.Constants.default_seed)  # Reset RNG
    mab_algo.partial_fit(
        decisions=df_extra_train_with_contexts[src.COLUMN_ITEM_ID],
        rewards=df_extra_train_with_contexts[src.COLUMN_RATING],
        contexts=contexts
    )

def group_interactions_by_user(interactions_df):
    interactions_by_user = interactions_df\
                        .groupby(src.COLUMN_USER_ID)[[src.COLUMN_ITEM_ID]]\
                        .apply(lambda df_user: df_user[src.COLUMN_ITEM_ID].tolist())\
                        .reset_index(name='interactions')
    interactions_by_user = interactions_by_user.reset_index(drop=True)
    return interactions_by_user

def create_contexts_list_user(interactions_df, users_embeddings):
    contexts = []

    for _, row in interactions_df.iterrows():
        user_id = row[src.COLUMN_USER_ID]
        contexts.append(users_embeddings[user_id][:users_embeddings.shape[1]])

    return contexts


class _LinearArmEncoded(_Linear):

    def __init__(self, rng: _BaseRNG, num_arms: int, n_jobs: int, backend: Optional[str],
                 alpha: Num, epsilon: Num, l2_lambda: Num, regression: str, scale: bool):
        super().__init__(rng, np.arange(num_arms).tolist(), n_jobs, backend, alpha, epsilon, l2_lambda, regression, scale)
        self.num_arms = num_arms
        self.users_temporal_infos = {}
    
    def _vectorized_predict_context(self, contexts: np.ndarray, is_predict: bool) -> List:

        arms = np.arange(self.num_arms)

        # Initializing array with expectations for each arm
        num_contexts = contexts.shape[0]
        arm_expectations = np.empty((num_contexts, self.num_arms), dtype=float)

        # With epsilon probability, assign random flag to context
        random_values = self.rng.rand(num_contexts)
        random_mask = np.array(random_values < self.epsilon)
        random_indices = random_mask.nonzero()[0]

        # For random indices, generate random expectations
        arm_expectations[random_indices] = self.rng.rand((random_indices.shape[0], self.num_arms))

        # For non-random indices, get expectations for each arm
        nonrandom_indices = np.where(~random_mask)[0]
        nonrandom_context = contexts[nonrandom_indices]

        arm_expectations[nonrandom_indices] = np.array([self.arm_to_model[arm].predict(nonrandom_context)
                                                        for arm in arms]).T

        return arm_expectations if len(arm_expectations) > 1 else arm_expectations[0]
    
    def fit(self, users_ids: np.ndarray, timestamps: np.ndarray, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> None:
        # users_ids e timestamps são parâmetros novos

        self.num_features = contexts.shape[1]
        for arm in self.arms:
            self.arm_to_model[arm].init(num_features=self.num_features)

        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

        # NOVO
        self.update_users_temporal_infos(users_ids, timestamps)
    

    # NOVO MÉTODO
    def update_users_temporal_infos(self, users_ids: np.ndarray, timestamps: np.ndarray) -> None:
        for user_id, timestamp in zip(users_ids, timestamps):
            if user_id not in self.users_temporal_infos:
                self.users_temporal_infos[user_id] = {
                    'timediffs': [],
                    'last_timestamp': timestamp,
                }
            else:
                self.users_temporal_infos[user_id]['timediffs'].append(timestamp - self.users_temporal_infos[user_id]['last_timestamp'])
                self.users_temporal_infos[user_id]['last_timestamp'] = timestamp

        for user_id in self.users_temporal_infos:
            if len(self.users_temporal_infos[user_id]['timediffs']) != 0:
                self.users_temporal_infos[user_id]['mean'] = np.array(self.users_temporal_infos[user_id]['timediffs']).mean()
                self.users_temporal_infos[user_id]['median'] = np.median(self.users_temporal_infos[user_id]['timediffs'])
            else:
                self.users_temporal_infos[user_id]['mean'] = 0
                self.users_temporal_infos[user_id]['median'] = 0
        

    
    def partial_fit(self, users_ids: np.ndarray, timestamps: np.ndarray, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> None:
        # Perform parallel fit
        self._parallel_fit(decisions, rewards, contexts)

        # NOVO
        self.update_users_temporal_infos(users_ids, timestamps)


from mabwiser.mab import MAB, LearningPolicyType, NeighborhoodPolicyType, NeighborhoodPolicy
from typing import List

from mabwiser._version import __author__, __copyright__, __email__, __version__
from mabwiser.approximate import _LSHNearest
from mabwiser.clusters import _Clusters
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _Linear
from mabwiser.neighbors import _KNearest, _Radius
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.treebandit import _TreeBandit
from mabwiser.ucb import _UCB1
from mabwiser.utils import Arm, Constants, check_true, create_rng

from typing import Union

class MABArmEncoded(MAB):
    def __init__(self,
                 num_arms: int,  # The list of arms
                 learning_policy: LearningPolicyType,  # The learning policy
                 neighborhood_policy: NeighborhoodPolicyType = None,  # The context policy, optional
                 seed: int = Constants.default_seed,  # The random seed
                 n_jobs: int = 1,  # Number of parallel jobs
                 backend: str = None  # Parallel backend implementation
                 ):
        """Initializes a multi-armed bandit (MAB) with the given arguments.

        Validates the arguments and raises exception in case there are violations.

        Parameters
        ----------
        arms : List[Union[int, float, str]]
            The list of all the arms available for decisions.
            Arms can be integers, strings, etc.
        learning_policy : LearningPolicyType
            The learning policy.
        neighborhood_policy : NeighborhoodPolicyType, optional
            The context policy. Default value is None.
        seed : numbers.Rational, optional
            The random seed to initialize the random number generator.
            Default value is set to Constants.default_seed.value
        n_jobs: int, optional
            This is used to specify how many concurrent processes/threads should be used for parallelized routines.
            Default value is set to 1.
            If set to -1, all CPUs are used.
            If set to -2, all CPUs but one are used, and so on.
        backend: str, optional
            Specify a parallelization backend implementation supported in the joblib library. Supported options are:
            - “loky” used by default, can induce some communication and memory overhead when exchanging input and
              output data with the worker Python processes.
            - “multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
            - “threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the
              called function relies a lot on Python objects.
            Default value is None. In this case the default backend selected by joblib will be used.

        Raises
        ------
        TypeError:  Arms were not provided in a list.
        TypeError:  Learning policy type mismatch.
        TypeError:  Context policy type mismatch.
        TypeError:  Seed is not an integer.
        TypeError:  Number of parallel jobs is not an integer.
        TypeError:  Parallel backend is not a string.
        TypeError:  For EpsilonGreedy, epsilon must be integer or float.
        TypeError:  For LinGreedy, epsilon must be an integer or float.
        TypeError:  For LinGreedy, l2_lambda must be an integer or float.
        TypeError:  For LinTS, alpha must be an integer or float.
        TypeError:  For LinTS, l2_lambda must be an integer or float.
        TypeError:  For LinUCB, alpha must be an integer or float.
        TypeError:  For LinUCB, l2_lambda must be an integer or float.
        TypeError:  For Softmax, tau must be an integer or float.
        TypeError:  For ThompsonSampling, binarizer must be a callable function.
        TypeError:  For UCB, alpha must be an integer or float.
        TypeError:  For LSHNearest, n_dimensions must be an integer or float.
        TypeError:  For LSHNearest, n_tables must be an integer or float.
        TypeError:  For LSHNearest, no_nhood_prob_of_arm must be None or List that sums up to 1.0.
        TypeError:  For Clusters, n_clusters must be an integer.
        TypeError:  For Clusters, is_minibatch must be a boolean.
        TypeError:  For Radius, radius must be an integer or float.
        TypeError:  For Radius, no_nhood_prob_of_arm must be None or List that sums up to 1.0.
        TypeError:  For KNearest, k must be an integer or float.

        ValueError: Invalid number of arms.
        ValueError: Invalid values (None, NaN, Inf) in arms.
        ValueError: Duplicate values in arms.
        ValueError: Number of parallel jobs is 0.
        ValueError: For EpsilonGreedy, epsilon must be between 0 and 1.
        ValueError: For LinGreedy, epsilon must be between 0 and 1.
        ValueError: For LinGreedy, l2_lambda cannot be negative.
        ValueError: For LinTS, alpha must be greater than zero.
        ValueError: For LinTS, l2_lambda must be greater than zero.
        ValueError: For LinUCB, alpha cannot be negative.
        ValueError: For LinUCB, l2_lambda cannot be negative.
        ValueError: For Softmax, tau must be greater than zero.
        ValueError: For UCB, alpha must be greater than zero.
        ValueError: For LSHNearest, n_dimensions must be gerater than zero.
        ValueError: For LSHNearest, n_tables must be gerater than zero.
        ValueError: For LSHNearest, if given, no_nhood_prob_of_arm list should sum up to 1.0.
        ValueError: For Clusters, n_clusters cannot be less than 2.
        ValueError: For Radius and KNearest, metric is not supported by scipy.spatial.distance.cdist.
        ValueError: For Radius, radius must be greater than zero.
        ValueError: For Radius, if given, no_nhood_prob_of_arm list should sum up to 1.0.
        ValueError: For KNearest, k must be greater than zero.
        """

        # Validate arguments
        # MAB._validate_mab_args(arms, learning_policy, neighborhood_policy, seed, n_jobs, backend)

        # Save the arguments
        self.arms = np.arange(num_arms)
        self.num_arms = num_arms
        self.seed = seed
        self.n_jobs = n_jobs
        self.backend = backend

        # Create the random number generator
        self._rng = create_rng(self.seed)
        self._is_initial_fit = False

        # Create the learning policy implementor
        lp = None
        if isinstance(learning_policy, LearningPolicy.EpsilonGreedy):
            lp = _EpsilonGreedy(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.epsilon)
        elif isinstance(learning_policy, LearningPolicy.Popularity):
            lp = _Popularity(self._rng, self.arms, self.n_jobs, self.backend)
        elif isinstance(learning_policy, LearningPolicy.Random):
            lp = _Random(self._rng, self.arms, self.n_jobs, self.backend)
        elif isinstance(learning_policy, LearningPolicy.Softmax):
            lp = _Softmax(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.tau)
        elif isinstance(learning_policy, LearningPolicy.ThompsonSampling):
            lp = _ThompsonSampling(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.binarizer)
        elif isinstance(learning_policy, LearningPolicy.UCB1):
            lp = _UCB1(self._rng, self.arms, self.n_jobs, self.backend, learning_policy.alpha)
        elif isinstance(learning_policy, LearningPolicy.LinGreedy):
            lp = _LinearArmEncoded(self._rng, num_arms, self.n_jobs, self.backend, 0, learning_policy.epsilon,
                         learning_policy.l2_lambda, "ridge", learning_policy.scale)
        elif isinstance(learning_policy, LearningPolicy.LinTS):
            lp = _LinearArmEncoded(self._rng, num_arms, self.n_jobs, self.backend, learning_policy.alpha, 0,
                         learning_policy.l2_lambda, "ts", learning_policy.scale)
        elif isinstance(learning_policy, LearningPolicy.LinUCB):
            lp = _LinearArmEncoded(self._rng, num_arms, self.n_jobs, self.backend, learning_policy.alpha, 0,
                         learning_policy.l2_lambda, "ucb", learning_policy.scale)
        else:
            check_true(False, ValueError("Undefined learning policy " + str(learning_policy)))

        # Create the mab implementor
        if neighborhood_policy:
            self.is_contextual = True

            # Do not use parallel fit or predict for Learning Policy when contextual
            lp.n_jobs = 1

            if isinstance(neighborhood_policy, NeighborhoodPolicy.Clusters):
                self._imp = _Clusters(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                      neighborhood_policy.n_clusters, neighborhood_policy.is_minibatch)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.LSHNearest):
                self._imp = _LSHNearest(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                        neighborhood_policy.n_dimensions, neighborhood_policy.n_tables,
                                        neighborhood_policy.no_nhood_prob_of_arm)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.KNearest):
                self._imp = _KNearest(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                      neighborhood_policy.k, neighborhood_policy.metric)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.Radius):
                self._imp = _Radius(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                    neighborhood_policy.radius, neighborhood_policy.metric,
                                    neighborhood_policy.no_nhood_prob_of_arm)
            elif isinstance(neighborhood_policy, NeighborhoodPolicy.TreeBandit):
                self._imp = _TreeBandit(self._rng, self.arms, self.n_jobs, self.backend, lp,
                                        neighborhood_policy.tree_parameters)
            else:
                check_true(False, ValueError("Undefined context policy " + str(neighborhood_policy)))
        else:
            self.is_contextual = isinstance(learning_policy, (LearningPolicy.LinGreedy, LearningPolicy.LinTS,
                                                              LearningPolicy.LinUCB))
            self._imp = lp
    
    def fit(self,
            users_ids: np.ndarray,
            timestamps: np.ndarray,
            decisions: Union[List[Arm], np.ndarray, pd.Series],  # Decisions that are made
            rewards: Union[List[Num], np.ndarray, pd.Series],  # Rewards that are received
            contexts: Union[None, List[List[Num]],
                            np.ndarray, pd.Series, pd.DataFrame] = None  # Contexts, optional
            ) -> None:
        """Fits the multi-armed bandit to the given *decisions*, their corresponding *rewards*
        and *contexts*, if any.

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
         contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame]
            The context under which each decision is made. Default value is ``None``, i.e., no contexts.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Decisions and rewards are not given as list, numpy array or pandas series.
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Length mismatch between decisions, rewards, and contexts.
        ValueError: Fitting contexts data when there is no contextual policy.
        ValueError: Contextual policy when fitting no contexts data.
        ValueError: Rewards contain ``None``, ``Nan``, or ``Infinity``.
        """

        # Validate arguments
        self._validate_fit_args(decisions, rewards, contexts)

        # Convert to numpy array for efficiency
        decisions = MAB._convert_array(decisions)
        rewards = MAB._convert_array(rewards)
        users_ids = MAB._convert_array(users_ids)
        timestamps = MAB._convert_array(timestamps)

        # Check rewards are valid
        check_true(np.isfinite(sum(rewards)), TypeError("Rewards cannot contain None, nan or infinity."))

        # Convert contexts to numpy array for efficiency
        contexts = self.__convert_context(contexts, decisions)

        # Call the fit method
        self._imp.fit(users_ids, timestamps, decisions, rewards, contexts)

        # Turn initial to true
        self._is_initial_fit = True

    def partial_fit(self,
                    users_ids: np.ndarray,
                    timestamps: np.ndarray,
                    decisions: Union[List[Arm], np.ndarray, pd.Series],
                    rewards: Union[List[Num], np.ndarray, pd.Series],
                    contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) -> None:
        """Updates the multi-armed bandit with the given *decisions*, their corresponding *rewards*
        and *contexts*, if any.

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
         contexts : Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] =
            The context under which each decision is made. Default value is ``None``, i.e., no contexts.

        Returns
        -------
        No return.

        Raises
        ------
        TypeError:  Decisions, rewards are not given as list, numpy array or pandas series.
        TypeError:  Contexts is not given as ``None``, list, numpy array, pandas series or data frames.

        ValueError: Length mismatch between decisions, rewards, and contexts.
        ValueError: Fitting contexts data when there is no contextual policy.
        ValueError: Contextual policy when fitting no contexts data.
        ValueError: Rewards contain ``None``, ``Nan``, or ``Infinity``
        """

        # Validate arguments
        self._validate_fit_args(decisions, rewards, contexts)

        # Convert to numpy array for efficiency
        decisions = MAB._convert_array(decisions)
        rewards = MAB._convert_array(rewards)
        users_ids = MAB._convert_array(users_ids)
        timestamps = MAB._convert_array(timestamps)

        # Check rewards are valid
        check_true(np.isfinite(sum(rewards)), TypeError("Rewards cannot contain None, NaN or infinity."))

        # Convert contexts to numpy array for efficiency
        contexts = self.__convert_context(contexts, decisions)

        # Call the fit or partial fit method
        if self._is_initial_fit:
            self._imp.partial_fit(users_ids, timestamps, decisions, rewards, contexts)
        else:
            self.fit(users_ids, timestamps, decisions, rewards, contexts)
    
    def __convert_context(self, contexts, decisions=None) -> Union[None, np.ndarray]:
        """
        Convert contexts to numpy array for efficiency.
        For fit and partial fit, decisions must be provided.
        The numpy array need to be in C row-major order for efficiency.
        """
        if contexts is None:
            return None
        elif isinstance(contexts, np.ndarray):
            if contexts.flags['C_CONTIGUOUS']:
                return contexts
            else:
                return np.asarray(contexts, order="C")
        elif isinstance(contexts, list):
            return np.asarray(contexts, order="C")
        elif isinstance(contexts, pd.DataFrame):
            if contexts.values.flags['C_CONTIGUOUS']:
                return contexts.values
            else:
                return np.asarray(contexts.values, order="C")
        elif isinstance(contexts, pd.Series):
            # When context is a series, we need to differentiate between
            # a single context with multiple features vs. multiple contexts with single feature
            is_called_from_fit = decisions is not None

            if is_called_from_fit:
                if len(decisions) > 1:  # multiple decisions exists
                    return np.asarray(contexts.values, order="C").reshape(-1, 1)  # go from 1D to 2D
                else:  # single decision
                    return np.asarray(contexts.values, order="C").reshape(1, -1)  # go from 1D to 2D

            else:  # For predictions, compare the shape to the stored context history

                # We need to find out the number of features (to distinguish Series shape)
                if isinstance(self.learning_policy, (LearningPolicy.LinGreedy,
                                                     LearningPolicy.LinTS,
                                                     LearningPolicy.LinUCB)):
                    first_arm = self.arms[0]
                    if isinstance(self._imp, _Linear):
                        num_features = self._imp.arm_to_model[first_arm].beta.size
                    else:
                        num_features = self._imp.contexts.shape[1]
                elif isinstance(self._imp, _TreeBandit):
                    # Even when fit() happened, the first arm might not necessarily have a fitted tree
                    # So we have to search for a fitted tree
                    for arm in self.arms:
                        try:
                            num_features = len(self._imp.arm_to_tree[arm].feature_importances_)
                        except:
                            continue
                else:
                    num_features = self._imp.contexts.shape[1]

                if num_features == 1:
                    return np.asarray(contexts.values, order="C").reshape(-1, 1)  # go from 1D to 2D
                else:
                    return np.asarray(contexts.values, order="C").reshape(1, -1)  # go from 1D to 2D

        else:
            raise NotImplementedError("Unsupported contexts data type")


from typing import Dict, List, Tuple, Union
from mabwiser.utils import Arm, Num, _BaseRNG
from scipy.special import expit

class BanditRecommenderArmEncoded(BanditRecommender):
    def _init(self, num_arms: int) -> None:
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
        self.mab = MABArmEncoded(num_arms, self.learning_policy, self.neighborhood_policy, self.seed, self.n_jobs, self.backend)
    
    def fit(self, users_ids: np.ndarray, timestamps: np.ndarray, decisions: Union[List[Arm], np.ndarray, pd.Series],
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
            self._init(np.unique(decisions).shape[0])
        self.mab.fit(users_ids, timestamps, decisions, rewards, contexts)
    
    def partial_fit(self, users_ids: np.ndarray, timestamps: np.ndarray, decisions: Union[List[Arm], np.ndarray, pd.Series],
                    rewards: Union[List[Num], np.ndarray, pd.Series],
                    contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None) -> None:
        
        self.mab.partial_fit(users_ids, timestamps, decisions, rewards, contexts)
    
    def generate_analyse(self, users_ids: np.ndarray, items_ids: np.ndarray, timestamps: np.ndarray, contexts: Union[None, List[List[Num]], np.ndarray, pd.Series, pd.DataFrame] = None, excluded_arms: List[List[Arm]] = None, apply_sigmoid: bool = True) \
            -> Union[Union[List[Arm], Tuple[List[Arm], List[Num]],
                     Union[List[List[Arm]], Tuple[List[List[Arm]], List[List[Num]]]]]]:
        self._validate_mab(is_fit=True)
        self._validate_get_rec(contexts, excluded_arms)

        # Get predicted expectations
        num_contexts = len(contexts) if contexts is not None else 1
        if num_contexts == 1:
            expectations = [self.mab.predict_expectations(contexts)]
        else:
            expectations = self.mab.predict_expectations(contexts)

        if apply_sigmoid:
            expectations = expit(expectations)

        # Create an exclusion mask, where exclusion_mask[context_ind][arm_ind] denotes if the arm with the
        # index arm_ind was excluded for context with the index context_ind.
        # The value will be True if it is excluded and those arms will not be returned as part of the results.
        arm_to_index = {arm: arm_ind for arm_ind, arm in enumerate(self.mab.arms)}
        exclude_mask = np.zeros((num_contexts, len(self.mab.arms)), dtype=bool)
        if excluded_arms is not None:
            for context_ind, excluded in enumerate(excluded_arms):
                exclude_mask[context_ind][[arm_to_index[arm] for arm in excluded if arm in arm_to_index]] = True

        # Set excluded item scores to -1, so they automatically get placed lower in best results
        expectations[exclude_mask] = -np.inf

        # Get best `top_k` results by sorting the expectations
        # NOVO, argsort completo
        arm_inds = np.argsort(expectations, axis=1)[:, ::-1]
        expectations = np.array([expectations[i][arm_inds[i]] for i in range(num_contexts)])

        # Coleta as informações temporais dos usuários
        cur_users_infos = {}
        for user_id, user_info in self.mab._imp.users_temporal_infos.items():
            cur_users_infos[user_id] = {
                'mean': user_info['mean'],
                'median': user_info['median'],
                'last_timestamp': user_info['last_timestamp'],
                'qnt_consumed_items': len(user_info['timediffs']) + 1
            }
        
        # Cria o DataFrame final da janela
        items_ranks = []
        timediffs = []
        mean_timediffs = []
        median_timediffs = []
        qnt_consumed_items = []
        first_item_score = []
        min_item_score = []
        q1_item_score = []
        q2_item_score = []
        q3_item_score = []
        max_item_score = []
        i = 0

        qnt_items = len(expectations[0])

        for user_id, item_id, timestamp in zip(users_ids, items_ids, timestamps):
            items_ranks.append(np.where(arm_inds[i] == item_id)[0][0])
            timediffs.append(timestamp - cur_users_infos[user_id]['last_timestamp'])
            mean_timediffs.append(cur_users_infos[user_id]['mean'])
            median_timediffs.append(cur_users_infos[user_id]['median'])
            qnt_consumed_items.append(cur_users_infos[user_id]['qnt_consumed_items'])

            scores = expectations[i]
            qnt_scores = qnt_items - cur_users_infos[user_id]['qnt_consumed_items']  # Quantidade de scores que não foram consumidos. Os scores consumidos possuem score -inf, então, eles não são considerados

            first_item_score.append(scores[0])
            min_item_score.append(scores[1])
            q1_item_score.append(scores[int(qnt_scores * 0.25)])
            q2_item_score.append(scores[int(qnt_scores * 0.5)])
            q3_item_score.append(scores[int(qnt_scores * 0.75)])
            max_item_score.append(scores[int(qnt_scores - 1)])

            cur_users_infos[user_id]['last_timestamp'] = timestamp
            i += 1
        
        df = pd.DataFrame({
            src.COLUMN_USER_ID: users_ids,
            src.COLUMN_ITEM_ID: items_ids,
            'rank': items_ranks,
            'timediff': timediffs,
            'mean_timediff': mean_timediffs,
            'median_timediff': median_timediffs,
            'qnt_consumed_items': qnt_consumed_items,
            'first_item_score': first_item_score,
            'min_item_score': min_item_score,
            'q1_item_score': q1_item_score,
            'q2_item_score': q2_item_score,
            'q3_item_score': q3_item_score,
            'max_item_score': max_item_score
        })
        
        return df

import os

FACTORS = 10

from scipy.sparse import csr_matrix


from src.algorithms.not_incremental.implicit.ImplicitRecommender import ImplicitRecommender

def test(df_full: pd.DataFrame, embeddings_generator: ImplicitRecommender, create_contexts: 'Callable[[pd.DataFrame, np.ndarray, np.ndarray], list[list[float]]]', save_path: str):

    dfs = []
    test_size = 0.5
    qnt_windows = 10

    df_full[src.COLUMN_USER_ID] = pd.factorize(df_full[src.COLUMN_USER_ID])[0]
    df_full[src.COLUMN_ITEM_ID] = pd.factorize(df_full[src.COLUMN_ITEM_ID])[0]

    split_index = int(len(df_full) * (1 - test_size))
    df_train = df_full[:split_index].copy()
    df_test_full = df_full[split_index:]

    df_test_full = df_test_full[(df_test_full[src.COLUMN_USER_ID].isin(df_train[src.COLUMN_USER_ID])) & (df_test_full[src.COLUMN_ITEM_ID].isin(df_train[src.COLUMN_ITEM_ID]))]
    df_test_full = df_test_full.reset_index(drop=True)

    print('Generating embeddings...')
    embeddings_generator.train(df_train)
    users_embeddings = embeddings_generator.users_embeddings
    items_embeddings = embeddings_generator.items_embeddings
    print('Generating contexts...')
    new_df_full = pd.concat([df_train, df_test_full])
    contexts = create_contexts(new_df_full, users_embeddings, items_embeddings)

    df_train['context']  = contexts[:len(df_train)]
    df_test_full['context'] = contexts[len(df_train):]

    df_tests_for_extra_train = []
    df_tests_for_eval = []

    for i in range(qnt_windows):
        start_index = int(len(df_test_full) * (i / qnt_windows))
        final_index = int(len(df_test_full) * ((i + 1) / qnt_windows))

        df_test_for_extra_train = df_test_full[start_index:final_index]

        df_tests_for_extra_train.append(df_test_for_extra_train)

        df_test_for_eval = df_test_for_extra_train[df_test_for_extra_train[src.COLUMN_RATING] == 1]
        df_test_for_eval = df_test_for_eval.reset_index(drop=True)

        df_tests_for_eval.append(df_test_for_eval)
    
    print(f'Training MAB...')
    model = BanditRecommenderArmEncoded(learning_policy=LearningPolicy.LinGreedy(epsilon=0), top_k=10)
    model.fit(df_train[src.COLUMN_USER_ID], df_train[src.COLUMN_TIMESTAMP], df_train[src.COLUMN_ITEM_ID], df_train[src.COLUMN_RATING], np.array(df_train['context'].tolist()))
    for i in range(qnt_windows):
        print(f"Loading window: {i+1}")

        current_df_train = pd.concat([df_train, *(df_tests_for_extra_train[:i])])
        interactions_by_user = group_interactions_by_user(current_df_train)

        df_test_for_evaluation = df_tests_for_eval[i]
        df_test_for_extra_train = df_tests_for_extra_train[i]

        filters = df_test_for_evaluation.merge(interactions_by_user, how='left', on=src.COLUMN_USER_ID)[['interactions']].values.squeeze(axis=1) 
 
        df = model.generate_analyse(df_test_for_evaluation[src.COLUMN_USER_ID], df_test_for_evaluation[src.COLUMN_ITEM_ID], df_test_for_evaluation[src.COLUMN_TIMESTAMP], np.array(df_test_for_evaluation['context'].tolist()), filters, apply_sigmoid=False)
        df['window'] = i+1
        dfs.append(df)

        model.partial_fit(df_test_for_extra_train[src.COLUMN_USER_ID], df_test_for_extra_train[src.COLUMN_TIMESTAMP], df_test_for_extra_train[src.COLUMN_ITEM_ID], df_test_for_extra_train[src.COLUMN_RATING], np.array(df_test_for_extra_train['context'].tolist()))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
  
    final_df = pd.concat(dfs)

    final_df.to_csv(f'{save_path}/final_df.csv', index=False)

    return final_df


# 1 - Correlação geral entre rank e timediff / mean_timediff ou timediff / median_timediff
def  calc_correlation(df, col1, col2):
    return df[[col1, col2]].corr()[col1][col2]


# 2 - Histograma e boxplot de correlação por usuário entre rank e timediff / mean_timediff ou timediff / median_timediff
def plot_correlation_by_user(df, col1, col2, save_code, save_path):
    corrs = []
    for user_id in df[src.COLUMN_USER_ID].unique():
        corrs.append(df[df[src.COLUMN_USER_ID] == user_id][[col1, col2]].corr()[col1][col2])
        if np.isnan(corrs[-1]):
            corrs.pop()  # Quando existe apenas uma interação de teste, a correlação é NaN

    plt.hist(corrs, bins=20)

    plt.title(f'Histogram: correlation between {col1} and {col2} by user')
    plt.xlabel('Correlation')
    plt.ylabel('Qnt. users')

    plt.savefig(f'{save_path}/{save_code}_hist_{col1}_{col2}.png')

    plt.clf()

    # plt.show()

    # Salva o gráfico
    plt.boxplot(corrs)

    plt.title(f'Boxplot: correlation between {col1} and {col2} by user')
    plt.ylabel('Correlation')

    plt.savefig(f'{save_path}/{save_code}_box_{col1}_{col2}.png')

    plt.clf()

    # plt.show()

# 3 - Describe (média, std, quartis, etc.) de item_rank para registros com timediff / current_user_mean_timediff > threshold e  current_user_mean_timediff < threshold (para vários valores de threshold)
def describe_by_threshold(df, col, lower_thresholds, upper_thresholds):
    describes = []

    for lower_threshold in lower_thresholds:
        describes.append(df[df[col] < lower_threshold]['rank'].describe())

    describes.append(df['rank'].describe())

    for upper_threshold in upper_thresholds:
        describes.append(df[df[col] > upper_threshold]['rank'].describe())

    final_df = pd.DataFrame(describes)
    final_df.index = [f'< {threshold}' for threshold in lower_thresholds] + ['all'] + [f'> {threshold}' for threshold in upper_thresholds]

    # display(final_df)

    return final_df


# 4 - Describe (média, std, quartis, etc.) da diferença entre o score do primeiro item e dos itens nas posições: segunda (min), primeiro quartil (Q1 — 25%), mediana (Q2 — 50%), terceiro quartil (Q3 — 75%) e último (max)
def describe_diffs(df):

    diffs_min = []
    diffs_q1 = []
    diffs_median = []
    diffs_q3 = []
    diffs_max = []

    for i in range(len(df)):
        current_row = df.iloc[i]
        first_score = current_row['first_item_score']

        diffs_min.append(first_score - current_row['min_item_score'])
        diffs_q1.append(first_score - current_row['q1_item_score'])
        diffs_median.append(first_score - current_row['q2_item_score'])
        diffs_q3.append(first_score - current_row['q3_item_score'])
        diffs_max.append(first_score - current_row['max_item_score'])

    final_df = pd.DataFrame({
        'first - second': diffs_min,
        'first - q1': diffs_q1,
        'first - median': diffs_median,
        'first - q3': diffs_q3,
        'first - max': diffs_max
    })

    final_df = final_df.describe().drop('count')

    # display(final_df)

    return final_df

# 5 - HitRate (HR) para registros com timediff / current_user_mean_timediff > threshold e  current_user_mean_timediff < threshold (para vários valores de threshold)
def hr_by_threshold(df, col, lower_thresholds, upper_thresholds, qnt_items):


    static_topn_list = [5, 10, 15, 20, 25, 50, 100, 250]
    percentage_topn_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]

    results = {}
    results['count'] = []
    for static_topn_value in static_topn_list:
        results[f'Top {static_topn_value}'] = []
    for percentage_topn_value in percentage_topn_list:
        results[f'Top {int(percentage_topn_value*100)}%'] = []
    
    def append_threshold_results(threshold, filter_type):
        if filter_type == 'lower':
            filtered_ranks = df[df[col] < threshold]['rank']
        elif filter_type == 'upper':
            filtered_ranks = df[df[col] > threshold]['rank']
        else:
            filtered_ranks = df['rank']
        count = len(filtered_ranks)
        results['count'].append(count)

        for static_topn_value in static_topn_list:
            results[f'Top {static_topn_value}'].append((filtered_ranks < static_topn_value).sum() / count)
        for percentage_topn_value in percentage_topn_list:
            results[f'Top {int(percentage_topn_value*100)}%'].append((filtered_ranks < (qnt_items * percentage_topn_value)).sum() / count)

    for lower_threshold in lower_thresholds:
        append_threshold_results(lower_threshold, filter_type='lower')

    append_threshold_results(None, None)

    for upper_threshold in upper_thresholds:
        append_threshold_results(upper_threshold, filter_type='upper')

    final_df = pd.DataFrame(results)
    final_df.index = [f'< {threshold}' for threshold in lower_thresholds] + ['all'] + [f'> {threshold}' for threshold in upper_thresholds]

    return final_df


from src.algorithms.not_incremental.implicit.ImplicitRecommender import ImplicitRecommender
from typing import Callable

def analyse_dataset(df_interactions: pd.DataFrame, embeddings_generator: ImplicitRecommender, context_builder: 'Callable[[pd.DataFrame, np.ndarray, np.ndarray], list[list[float]]]', save_path: str, processed_save_path: str, min_qnt_interactions: int):

    os.makedirs(save_path, exist_ok=True)
    
    if not os.path.exists(os.path.join(processed_save_path, 'final_df.csv')):
        df = test(df_interactions, embeddings_generator, context_builder, processed_save_path)
    else:
        print('Loading processed DataFrame...')
        df = pd.read_csv(os.path.join(processed_save_path, 'final_df.csv'))

    qnt_items = df_interactions[:int(len(df_interactions)*0.5)][src.COLUMN_ITEM_ID].nunique()

    # Filter interactions by user that has < min_qnt_interactions
    users_interactions = df[src.COLUMN_USER_ID].value_counts()
    users_interactions = users_interactions[users_interactions >= min_qnt_interactions].index
    df = df[df[src.COLUMN_USER_ID].isin(users_interactions)]

    df['mean_timediff_frac'] = df['timediff'] / df['mean_timediff']
    df['median_timediff_frac'] = df['timediff'] / df['median_timediff']

    # 1.1 - Correlação geral entre rank e timediff / mean_timediff
    print('1.1 - Loading correlation between rank and timediff / mean_timediff...')
    corr = calc_correlation(df, 'rank', 'mean_timediff_frac')
    with open(f'{save_path}/1_1-corr_rank_mean_timediff.txt', 'w') as f:
        f.write(str(corr))
    
    # 1.2 - Correlação geral entre rank e timediff / median_timediff
    print('1.2 - Loading correlation between rank and timediff / median_timediff...')
    corr = calc_correlation(df, 'rank', 'median_timediff_frac')
    with open(f'{save_path}/1_2-corr_rank_median_timediff.txt', 'w') as f:
        f.write(str(corr))

    # 2.1 - Histograma e boxplot de correlação por usuário entre rank e timediff / mean_timediff
    print('2.1 - Loading correlation by user between rank and timediff / mean_timediff...')
    plot_correlation_by_user(df, 'rank', 'mean_timediff_frac', '2_1', save_path)

    # 2.2 - Histograma de correlação por usuário entre rank e timediff / median_timediff
    print('2.2 - Loading correlation by user between rank and timediff / median_timediff...')
    plot_correlation_by_user(df, 'rank', 'median_timediff_frac', '2_2', save_path)

    # 3.1 - Describe (média, std, quartis, etc.) de item_rank para registros com timediff / current_user_mean_timediff > threshold e  current_user_mean_timediff < threshold (para vários valores de threshold)
    upper_thresholds = [1.25, 1.5, 1.75, 2, 2.5, 3, 5, 10]
    lower_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75]

    print('3.1 - Loading describe of item_rank for timediff / current_user_mean_timediff > threshold and current_user_mean_timediff < threshold...')
    desc_df = describe_by_threshold(df, 'mean_timediff_frac', lower_thresholds, upper_thresholds)
    desc_df.to_csv(f'{save_path}/3_1-desc_mean_timediff_frac.csv')
    desc_df.to_markdown(f'{save_path}/3_1-desc_mean_timediff_frac.md')

    # 3.2 - Describe (média, std, quartis, etc.) de item_rank para registros com timediff / current_user_median_timediff > threshold e  current_user_median_timediff < threshold (para vários valores de threshold)
    print('3.2 - Loading describe of item_rank for timediff / current_user_median_timediff > threshold and current_user_median_timediff < threshold...')
    desc_df = describe_by_threshold(df, 'median_timediff_frac', lower_thresholds, upper_thresholds)
    desc_df.to_csv(f'{save_path}/3_2-desc_median_timediff_frac.csv')
    desc_df.to_markdown(f'{save_path}/3_2-desc_median_timediff_frac.md')

    # 4 - Describe (média, std, quartis, etc.) da diferença entre o score do primeiro item e dos itens nas posições: segunda (min), primeiro quartil (Q1 — 25%), mediana (Q2 — 50%), terceiro quartil (Q3 — 75%) e último (max)
    print('4 - Loading describe of the difference between the score of the first item and the items in the positions...')
    desc_df = describe_diffs(df)
    desc_df.to_csv(f'{save_path}/4-desc_diffs.csv')
    desc_df.to_markdown(f'{save_path}/4-desc_diffs.md')

    # 5.1 - HitRate (HR) para registros com timediff / current_user_mean_timediff > threshold e  current_user_mean_timediff < threshold (para vários valores de threshold)
    print('5.1 - Loading HR for timediff / current_user_mean_timediff > threshold and current_user_mean_timediff < threshold...')
    hr_df = hr_by_threshold(df, 'mean_timediff_frac', lower_thresholds, upper_thresholds, qnt_items)
    hr_df.to_csv(f'{save_path}/5_1-hr_mean_timediff_frac.csv')
    hr_df.to_markdown(f'{save_path}/5_1-hr_mean_timediff_frac.md')

    # 5.2 - HitRate (HR) para registros com timediff / current_user_median_timediff > threshold e  current_user_median_timediff < threshold (para vários valores de threshold)
    print('5.2 - Loading HR for timediff / current_user_median_timediff > threshold and current_user_median_timediff < threshold...')
    hr_df = hr_by_threshold(df, 'median_timediff_frac', lower_thresholds, upper_thresholds, qnt_items)
    hr_df.to_csv(f'{save_path}/5_2-hr_median_timediff_frac.csv')
    hr_df.to_markdown(f'{save_path}/5_2-hr_median_timediff_frac.md')
