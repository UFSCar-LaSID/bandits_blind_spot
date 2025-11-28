import numpy as np
import pandas as pd
from collections import defaultdict

import src
from src.algorithms.incremental.mab2rec.Mab2recRecommender import Mab2RecRecommender
from src.scripts.utils.Logger import Logger
from datetime import datetime, timedelta

class TimeAwareLinBoltzmann(Mab2RecRecommender):
    """
    Time-Aware Linear Bandit with Boltzmann exploration, returning Top-K recommendations.
    """

    def __init__(
        self,
        user_column: str = src.COLUMN_USER_ID,
        item_column: str = src.COLUMN_ITEM_ID,
        rating_column: str = src.COLUMN_RATING,
        logger: Logger = None,
        hyperparameters: dict = None
    ):
        super().__init__(user_column, item_column, rating_column, logger)
        self.hyperparameters = hyperparameters or {}

        # Top-K (Top-N) 
        self.top_k = src.TOP_N

        # Bandit parameters
        self.item_id_to_arm = {}
        self.A = None
        self.b = None
        self.A_inv = None
        self.theta_matrix = None
        self.d = None

        # Track consumed items by each user 
        self.consumed_items_by_user = defaultdict(set)

        self.interactions_by_user = None

        # Time dict 
        self.user_avg_tdelta = {}
        self.user_count_inter = {}
        self.user_last_tdelta = {}

        # Time-based hyperparams
        self.rho = self.hyperparameters.get("rho", 1.0)
        self.temperature_method = self.hyperparameters.get("temperature_method", "basic_time")
        self.alpha = self.hyperparameters.get("alpha", 0.1)
        self.H_min = self.hyperparameters.get("H_min", 0.5)
        self.H_max = self.hyperparameters.get("H_max", 1.5)

    def train(self, interactions_df: pd.DataFrame, contexts):
        if self.logger:
            self.logger.print_and_log("TimeAwareLinBoltzmann: train called.")

        self.interactions_by_user = self._Mab2RecRecommender__group_interactions_by_user(interactions_df)

        # Map each unique item to an arm index
        unique_items = interactions_df[self.item_column].unique()
        self.item_id_to_arm = {item_id: idx for idx, item_id in enumerate(unique_items)}
        num_arms = len(self.item_id_to_arm)

        # Determine context dimension d
        if isinstance(contexts, np.ndarray):
            self.d = contexts.shape[1]
        else:
            self.d = len(contexts[0]) if len(contexts) > 0 else 0

        # Initialize bandit parameters
        self.A = [np.identity(self.d) for _ in range(num_arms)]
        self.b = [np.zeros(self.d) for _ in range(num_arms)]
        self.A_inv = [np.identity(self.d) for _ in range(num_arms)]

        # Single 2D matrix of shape (num_arms, d) for all thetas
        self.theta_matrix = np.zeros((num_arms, self.d))

        # Reset time dictionaries
        self.user_avg_tdelta.clear()
        self.user_count_inter.clear()
        self.user_last_tdelta.clear()

        # Fill consumed items dictionary
        for row in interactions_df.itertuples():
            user_id = getattr(row, self.user_column)
            item_id = getattr(row, self.item_column)
            self.consumed_items_by_user[user_id].add(item_id)

        # Go through the interactions
        decisions = interactions_df[self.item_column].values
        rewards = interactions_df[self.rating_column].values
        user_ids = interactions_df[self.user_column].values
        timediffs = (interactions_df[src.COLUMN_TIMEDIFF].values
                     if src.COLUMN_TIMEDIFF in interactions_df.columns
                     else np.ones(len(interactions_df)))

        n = min(len(decisions), len(contexts))
        for i in range(n):
            item_id = decisions[i]
            reward = rewards[i]
            user_id = user_ids[i]
            tdiff = timediffs[i]
            arm_idx = self.item_id_to_arm[item_id]
            x = contexts[i]

            # Update bandit parameters via Sherman–Morrison
            self._update_bandit(arm_idx, reward, x)
            # Update user time stats
            self._update_user_time(user_id, tdiff)

    def partial_train(self, interactions_df: pd.DataFrame, contexts):
        if self.logger:
            self.logger.print_and_log("TimeAwareLinBoltzmann: partial_train called.")

        # Merge interactions data for pipeline integrity
        new_interactions = self._Mab2RecRecommender__group_interactions_by_user(interactions_df)
        if self.interactions_by_user is not None:
            self.interactions_by_user = self._Mab2RecRecommender__merge_interactions_by_user(
                self.interactions_by_user, new_interactions
            )
        else:
            self.interactions_by_user = new_interactions

        # Also update our dictionary of consumed items
        for row in interactions_df.itertuples():
            user_id = getattr(row, self.user_column)
            item_id = getattr(row, self.item_column)
            self.consumed_items_by_user[user_id].add(item_id)

        # Retrieve arrays
        decisions = interactions_df[self.item_column].values
        rewards = interactions_df[self.rating_column].values
        user_ids = interactions_df[self.user_column].values
        timediffs = (interactions_df[src.COLUMN_TIMEDIFF].values
                     if src.COLUMN_TIMEDIFF in interactions_df.columns
                     else np.ones(len(interactions_df)))

        n = min(len(decisions), len(contexts))
        for i in range(n):
            item_id = decisions[i]
            reward = rewards[i]
            user_id = user_ids[i]
            tdiff = timediffs[i]

            if item_id not in self.item_id_to_arm:
                continue

            arm_idx = self.item_id_to_arm[item_id]
            x = contexts[i]
            # Sherman–Morrison bandit update
            self._update_bandit(arm_idx, reward, x)
            # Time stats update
            self._update_user_time(user_id, tdiff)

    def recommend(self, users_ids: 'list[int]', contexts) -> 'tuple[list[list[int]], list[list[float]]]':
        if self.logger:
            self.logger.print_and_log(f"TimeAwareLinBoltzmann: recommend called for {len(users_ids)} users.")
        start_time = datetime.now()

        all_arms = list(self.item_id_to_arm.values())
        arm_idx_to_item_id = {v: k for k, v in self.item_id_to_arm.items()}

        recommended_item_ids_per_user = []
        recommended_scores_per_user = []

        time_spent_init = datetime.now() - start_time

        times_spent_scores = []
        times_spent_probs = []
        times_spent_generating_rec_list = []
        times_spent_sample_topk = []
        # Precompute arm variances once outside the loop

        start_variance= datetime.now()
        arm_variances = np.var(self.theta_matrix, axis=1)
        time_spent_variance = datetime.now() - start_variance

        for user_idx, user_id in enumerate(users_ids):
            if user_idx >= len(contexts):
                recommended_item_ids_per_user.append([])
                recommended_scores_per_user.append([])
                continue

            user_context_vector = contexts[user_idx]
            # Vectorized scores for all arms: shape (num_arms,)
            start_score = datetime.now()
            scores = self.theta_matrix.dot(user_context_vector)
            times_spent_scores.append(datetime.now() - start_score)

            # Compute base temperature
            delta_t = self.user_last_tdelta.get(user_id, 1.0)
            delta_m = self.user_avg_tdelta.get(user_id, 1.0)
            T_base = self._compute_base_temperature(delta_t, delta_m)

            # Apply variance and entropy adjustments if enabled
            start_probs = datetime.now()
            # Now compute probabilities
            if self.hyperparameters.get("use_variance_entropy", True):
                # Arm-wise temperature = T_base * (1.0 + alpha * variance)
                T_i = T_base * (1.0 + self.alpha * arm_variances)
                probs = self._compute_probs_with_armwise_temp(scores, T_i)

                # Compute entropy
                H = -np.sum(probs * np.log(probs + 1e-12))
                if H > self.H_max:
                    T_i *= 0.9
                elif H < self.H_min:
                    T_i *= 1.1

                probs = self._compute_probs_with_armwise_temp(scores, T_i)
            else:
                T = max(T_base, 1e-6)
                exp_vals = np.exp(scores / T)
                probs = exp_vals / (exp_vals.sum() + 1e-12)
            times_spent_probs.append(datetime.now() - start_probs)

            # Filter out consumed items by setting probability to -1
            consumed_set = self.consumed_items_by_user.get(user_id, set())
            for arm_idx in all_arms:
                item_id = arm_idx_to_item_id[arm_idx]
                if item_id in consumed_set:
                    probs[arm_idx] = -1.0

            # Re-normalize any negative probs to zero
            local_probs = np.copy(probs)
            local_probs[local_probs < 0.0] = 0.0
            sum_probs = local_probs.sum()
            if sum_probs == 0.0:
                # Edge case: everything consumed or all zero
                recommended_item_ids_per_user.append([])
                recommended_scores_per_user.append([])
                continue
            local_probs /= sum_probs

            # Sample Top-K recommendations
            start_topk = datetime.now()
            # Sample Top-K arms from the distribution
            all_arms_array = np.array(all_arms)
            np.random.seed(42)
            chosen_arms = np.random.choice(all_arms_array, size=self.top_k, replace=False, p=local_probs)
            times_spent_sample_topk.append(datetime.now() - start_topk)

            start_generation_recs_list = datetime.now()
            chosen_probs = [local_probs[arm] for arm in chosen_arms]
            chosen_item_ids = [arm_idx_to_item_id[idx] for idx in chosen_arms]

            recommended_item_ids_per_user.append(chosen_item_ids)
            recommended_scores_per_user.append(chosen_probs)
            times_spent_generating_rec_list.append(datetime.now() - start_generation_recs_list)

        total_time_spent = datetime.now() - start_time

        time_spent_scores = timedelta(0)
        time_spent_probs = timedelta(0)
        time_spent_generating_rec_list = timedelta(0)
        time_spent_topk = timedelta(0)
        for i in range(len(times_spent_generating_rec_list)):
            time_spent_scores += times_spent_scores[i]
            time_spent_probs += times_spent_probs[i]
            time_spent_generating_rec_list += times_spent_generating_rec_list[i]
            time_spent_topk += times_spent_sample_topk[i]

        self.logger.print_and_log(f'Total time spent: {total_time_spent}')
        self.logger.print_and_log(f'Time init: {time_spent_init} (~{(time_spent_init / total_time_spent) * 100:.2f} %)')
        self.logger.print_and_log(f'Time scores: {time_spent_scores} (~{(time_spent_scores / total_time_spent) * 100:.2f} %)')
        self.logger.print_and_log(f'Time variance: {time_spent_variance} (~{(time_spent_variance / total_time_spent) * 100:.2f} %)')
        self.logger.print_and_log(f'Time probs: {time_spent_probs} (~{(time_spent_probs / total_time_spent) * 100:.2f} %)')
        self.logger.print_and_log(f'Time sample topk: {time_spent_topk} (~{(time_spent_topk / total_time_spent) * 100:.2f} %)')
        self.logger.print_and_log(f'Time generating rec list: {time_spent_generating_rec_list} (~{(time_spent_generating_rec_list / total_time_spent) * 100:.2f} %)')

        return recommended_item_ids_per_user, recommended_scores_per_user

    def _update_bandit(self, arm_idx: int, reward: float, x: np.ndarray):
        """
        Updates A_inv, b, and theta_matrix for arm_idx using the Sherman–Morrison formula
        to avoid full matrix inversion on each interaction.
        """
        # A[arm] -> A[arm] + x x^T
        # A_inv_new = A_inv_old - (A_inv_old x x^T A_inv_old) / (1 + x^T A_inv_old x)

        A_inv_old = self.A_inv[arm_idx]
        Ax = A_inv_old @ x
        denominator = 1.0 + x.T @ Ax

        # Rank-1 update for A_inv
        self.A_inv[arm_idx] = A_inv_old - np.outer(Ax, Ax) / denominator

        self.A[arm_idx] += np.outer(x, x)

        # Update b and then recompute the theta vector for this arm
        self.b[arm_idx] += reward * x
        self.theta_matrix[arm_idx] = self.A_inv[arm_idx].dot(self.b[arm_idx])

    def _update_user_time(self, user_id: int, tdiff: float):
        if user_id not in self.user_count_inter:
            self.user_count_inter[user_id] = 0
            self.user_avg_tdelta[user_id] = 0.0

        count = self.user_count_inter[user_id]
        old_avg = self.user_avg_tdelta[user_id]
        new_avg = (old_avg * count + tdiff) / (count + 1)
        self.user_count_inter[user_id] = count + 1
        self.user_avg_tdelta[user_id] = new_avg
        self.user_last_tdelta[user_id] = tdiff

    def _compute_probs_with_armwise_temp(self, scores: np.ndarray, T_array: np.ndarray):
        T_array = np.maximum(T_array, 1e-6)
        clipped_scores = scores / T_array
        clipped_scores = np.clip(clipped_scores, -100.0, 100.0)
        exp_vals = np.exp(clipped_scores)
        probs = exp_vals / (exp_vals.sum() + 1e-12)
        return probs

    def _compute_base_temperature(self, delta_t: float, delta_m: float) -> float:
        """Compute the base temperature using different methods."""
        method = self.hyperparameters.get("temperature_method", "traditional")
        if delta_m <= 0:
            delta_m = 1.0  # Avoid division by zero

        rho = self.rho
        T_fixed = self.hyperparameters.get("temperature_fixed", 0.1)

        if method == 'fixed':
            return T_fixed
        elif method == 'delta_t':
            return delta_t * rho
        elif method == 'traditional':
            return (delta_t / delta_m) * rho
        elif method == 'greg':
            return (2 * ((delta_t / delta_m) * rho)) ** 2
        elif method == "log_scale":
            return np.log(1 + delta_t / delta_m) * rho
        elif method == "fractional_power":
            return ((delta_t / delta_m) ** 0.5) * rho
        elif method == "exponential":
            return np.exp(delta_t / delta_m) ** 2
        else:
            raise ValueError(f"Unknown temperature method: {method}")
