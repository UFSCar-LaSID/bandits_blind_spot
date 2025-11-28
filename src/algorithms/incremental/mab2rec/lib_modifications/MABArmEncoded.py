# Code adapted from: https://github.com/fidelity/mabwiser/blob/master/mabwiser/mab.py
# The difference is that the original code accept different types of arms (strings, integers, etc) and the modified code only accept sequential integers as arms. With this modification, the code can be optimized.

from mabwiser.mab import MAB, LearningPolicyType, NeighborhoodPolicyType

from mabwiser.utils import Constants, check_true, create_rng
from mab2rec import LearningPolicy
from mab2rec import NeighborhoodPolicy
import numpy as np
from mabwiser.neighbors import _KNearest, _Radius
from mabwiser.treebandit import _TreeBandit
from mabwiser.clusters import _Clusters
from mabwiser.approximate import _LSHNearest

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

from src.algorithms.incremental.mab2rec.lib_modifications.LinearArmEncoded import _LinTSOptimized, _LinUCBOptimized, _RidgeRegressionOptimized, LinearArmEncodedOptimized

class MABArmEncodedOptimized(MAB):
    def __init__(self,
                 num_arms: int,  # The list of arms
                 num_features: int,
                 learning_policy: LearningPolicyType,  # The learning policy
                 neighborhood_policy: NeighborhoodPolicyType = None,  # The context policy, optional
                 seed: int = Constants.default_seed,  # The random seed
                 n_jobs: int = 1,  # Number of parallel jobs
                 backend: str = None,  # Parallel backend implementation
                 device: str = 'cpu'
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
        self.num_features = num_features
        self.seed = seed
        self.n_jobs = n_jobs
        self.backend = backend
        self.device = device

        # Create the random number generator
        self._rng = create_rng(self.seed)
        self._is_initial_fit = False

        # Create the learning policy implementor
        lp = None
        if isinstance(learning_policy, LearningPolicy.LinGreedy):
            lp = LinearArmEncodedOptimized(self._rng, num_arms, self.num_features, self.n_jobs, self.backend, 0, learning_policy.epsilon,
                         learning_policy.l2_lambda, "ridge", learning_policy.scale, self.device)
        elif isinstance(learning_policy, LearningPolicy.LinTS):
            lp = LinearArmEncodedOptimized(self._rng, num_arms, self.num_features, self.n_jobs, self.backend, learning_policy.alpha, 0,
                         learning_policy.l2_lambda, "ts", learning_policy.scale, self.device)
        elif isinstance(learning_policy, LearningPolicy.LinUCB):
            lp = LinearArmEncodedOptimized(self._rng, num_arms, self.num_features, self.n_jobs, self.backend, learning_policy.alpha, 0,
                         learning_policy.l2_lambda, "ucb", learning_policy.scale, self.device)
        else:
            check_true(False, ValueError("Undefined learning policy " + str(learning_policy)))

        if neighborhood_policy:
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
            self._imp = lp
        
        self.is_contextual = True
    
    @property
    def learning_policy(self):
        """
        Creates named tuple of the learning policy based on the implementor.

        Returns
        -------
        The learning policy.

        Raises
        ------
        NotImplementedError: MAB learning_policy property not implemented for this learning policy.

        """
        if isinstance(self._imp, (_LSHNearest, _KNearest, _Radius, _TreeBandit)):
            lp = self._imp.lp
        elif isinstance(self._imp, _Clusters):
            lp = self._imp.lp_list[0]
        else:
            lp = self._imp

        if isinstance(lp, _EpsilonGreedy):
            if issubclass(type(lp), _Popularity):
                return LearningPolicy.Popularity()
            else:
                return LearningPolicy.EpsilonGreedy(lp.epsilon)
        elif isinstance(lp, _Linear):
            if lp.regression == 'ridge':
                return LearningPolicy.LinGreedy(lp.epsilon, lp.l2_lambda, lp.scale)
            elif lp.regression == 'ts':
                return LearningPolicy.LinTS(lp.alpha, lp.l2_lambda, lp.scale)
            elif lp.regression == 'ucb':
                return LearningPolicy.LinUCB(lp.alpha, lp.l2_lambda, lp.scale)
            else:
                check_true(False, ValueError("Undefined regression " + str(lp.regression)))
        elif isinstance(lp, LinearArmEncodedOptimized):
            if lp.regression == 'ridge':
                return _RidgeRegressionOptimized(lp.rng, lp.alpha, lp.l2_lambda, lp.scale, self.device)
            elif lp.regression == 'ts':
                return _LinTSOptimized(lp.alpha, lp.l2_lambda, lp.scale, self.device)
            elif lp.regression == 'ucb':
                return _LinUCBOptimized(lp.alpha, lp.l2_lambda, lp.scale, self.device)
            else:
                check_true(False, ValueError("Undefined regression " + str(lp.regression)))
        elif isinstance(lp, _Random):
            return LearningPolicy.Random()
        elif isinstance(lp, _Softmax):
            return LearningPolicy.Softmax(lp.tau)
        elif isinstance(lp, _ThompsonSampling):
            return LearningPolicy.ThompsonSampling(lp.binarizer)
        elif isinstance(lp, _UCB1):
            return LearningPolicy.UCB1(lp.alpha)
        else:
            raise NotImplementedError("MAB learning_policy property not implemented for this learning policy.")