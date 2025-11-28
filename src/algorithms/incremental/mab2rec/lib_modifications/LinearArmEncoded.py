# Code adapted from: https://github.com/fidelity/mabwiser/blob/master/mabwiser/linear.py
# The difference is that the original code accept different types of arms (strings, integers, etc) and the modified code only accept sequential integers as arms. With this modification, the code can be optimized.

from mabwiser.utils import Arm, Num, _BaseRNG
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import src

class _RidgeRegressionOptimized:

    def __init__(self, rng: _BaseRNG, alpha: Num = 1.0, l2_lambda: Num = 1.0, scale: bool = False, device: str = 'cpu'):
        self.rng = rng                      
        self.alpha = alpha                 
        self.l2_lambda = l2_lambda          
        self.scale = scale                 

        self.beta = None                    
        self.A = None                       
        self.A_inv = None                   
        self.Xty = None
        self.scaler = None
        self.device = device

    def init(self, num_features: int, num_arms: int):
        self.Xty = torch.zeros((num_arms, num_features), device=self.device, dtype=torch.double)
        self.A = torch.eye(num_features, device=self.device, dtype=torch.double).unsqueeze(0).repeat(num_arms, 1, 1) * self.l2_lambda
        #self.A_inv = self.A.clone()
        self.beta = torch.zeros((num_arms, num_features), device=self.device, dtype=torch.double)

    def fit(self, decisions: np.ndarray, X: np.ndarray, y: np.ndarray):

        X_device = torch.tensor(X, device=self.device)
        y_device = torch.tensor(y, device=self.device)
        decisions_device = torch.tensor(decisions, device=self.device)

        self.A.index_add_(0, decisions_device, torch.einsum('ni,nj->nij', X_device, X_device))

        self.Xty.index_add_(0, decisions_device, X_device * y_device.view(-1, 1))

        for j in range(0, self.beta.shape[0], src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE):            
            self.beta[j:j+src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE] = torch.linalg.solve(
                self.A[j:j+src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE],
                self.Xty[j:j+src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE]
            )

    def predict(self, x: np.ndarray):
        return torch.matmul(torch.tensor(x, device=self.device, dtype=torch.double), self.beta.T)


class _LinTSOptimized(_RidgeRegressionOptimized):
    
    def predict(self, x: np.ndarray):
        x_torch = torch.tensor(x, device=self.device, dtype=torch.double)

        num_arms, num_features = self.beta.shape
        num_contexts = x.shape[0]

        scores = torch.empty((num_contexts, num_arms), device=self.device, dtype=torch.double)

        eps = torch.from_numpy(self.rng.standard_normal(size=(num_contexts, num_features))).to(device=self.device, dtype=torch.double)

        for start in range(0, num_arms, src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE):
            end = min(start + src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE, num_arms)  

            beta_chunk = self.beta[start:end]
            A_chunk = self.A[start:end]
            A_inv_chunk = torch.linalg.inv(A_chunk)

            L_chunk = torch.linalg.cholesky((self.alpha ** 2) * A_inv_chunk)
            beta_sampled = torch.einsum('bd,add->bad', eps, L_chunk) + beta_chunk

            scores[:, start:end] = torch.einsum('bd,bad->ba', x_torch, beta_sampled)

        return scores  # shape: [B, M]


class _LinUCBOptimized(_RidgeRegressionOptimized):

    def predict(self, x: np.ndarray):

        x = torch.tensor(x, device=self.device)

        scores = torch.matmul(x, self.beta.T)

        for j in range(0, self.beta.shape[0], src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE):
            x_A_inv = torch.matmul(x, torch.linalg.inv(self.A[j: j+src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE]))

            ucb = self.alpha * torch.sqrt(torch.sum(x_A_inv * x, axis=2))

            scores[:, j: j+src.INCREMENTAL_ALGO_ITEMS_BATCH_SIZE] += ucb.T
        
        return scores

class LinearArmEncodedOptimized:
    factory = {
        "ts": _LinTSOptimized, 
        "ucb": _LinUCBOptimized, 
        "ridge": _RidgeRegressionOptimized
    }

    def __init__(self, rng: _BaseRNG, num_arms: int, num_features:int, n_jobs: int, backend: Optional[str],
                 alpha: Num, epsilon: Num, l2_lambda: Num, regression: str, scale: bool, device: str):
        self.alpha = alpha
        self.epsilon = epsilon
        self.l2_lambda = l2_lambda
        self.regression = regression
        self.scale = scale
        self.n_jobs = n_jobs
        self.backend = backend
        self.rng = rng
        self.num_arms = num_arms
        self.num_features = num_features
        self.device = device
        self.model = self.factory[regression](rng, alpha, l2_lambda, scale, device)
        self.model.init(self.num_features, self.num_arms)

    def _vectorized_predict_context(self, contexts: np.ndarray, is_predict: bool) -> List:

        num_contexts = contexts.shape[0]
        arm_expectations = self.model.predict(contexts)

        random_values = self.rng.rand(num_contexts)
        random_mask = np.array(random_values < self.epsilon)
        random_indices = random_mask.nonzero()[0]

        arm_expectations[random_indices] = torch.tensor(self.rng.rand((random_indices.shape[0], self.num_arms)), device=self.device)

        return arm_expectations if len(arm_expectations) > 1 else arm_expectations[0]
    
    def fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> None:
        self._fit(decisions, rewards, contexts)
    
    def partial_fit(self, decisions: np.ndarray, rewards: np.ndarray, contexts: np.ndarray = None) -> None:
        self._fit(decisions, rewards, contexts)
    
    def _fit(self, decisions: np.ndarray, rewards: np.ndarray,
                      contexts: Optional[np.ndarray] = None):
        self.model.fit(decisions, contexts, rewards)
    
    def predict_expectations(self, contexts: np.ndarray = None) -> Union[Dict[Arm, Num], List[Dict[Arm, Num]]]:
        return self._vectorized_predict_context(contexts, is_predict=False)