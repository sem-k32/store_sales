""" every trainer is specific (relative to models, data and other features)
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from typing import Callable, Optional
from tqdm import tqdm

from .regression import PoissonRegr


class Trainer:
    def __init__(
            self, 
            regression: PoissonRegr,
            train_data: pd.DataFrame,
            optimizer: optim.Optimizer,
            lr_sched: Optional[LRScheduler],
            projector,
            max_epochs: int,
            grad_tol: float
        ) -> None:
        """ML-optimizer. If lambda model is linear, then liklyhood is convex. 
            Train on the whole given dataset.

        Args:
            regression (PoissonRegr): _description_
            train_data (pd.DataFrame): _description_
            optimizer (optim.Optimizer): _description_
            lr_sched (LRScheduler): _description_
            projector: _description_
            max_epochs (int): _description_
            grad_tol (float): _description_
        """
        self._regression = regression
        self._train_data = train_data
        self._optimizer = optimizer
        self._lr_sched = lr_sched
        self._projector = projector
        self._max_epochs = max_epochs
        self._grad_tol = grad_tol

    def _paramsNormGradient(self) -> float:
        lambda_spline = self._regression._lambd_func
        with torch.no_grad():
            grad_norm = 0
            for param in lambda_spline.parameters():
                grad_norm += torch.sum(param.grad ** 2)

            return torch.sqrt(grad_norm).item()

    def Train(self, epoch_callback: Callable) -> None:
        epoch_iter = tqdm(range(self._max_epochs), desc="Loss: ", leave=False)
        for epoch in epoch_iter:
            self._optimizer.zero_grad()
            neg_ln_lk = self._regression.negLnLiklyhood(self._train_data)
            neg_ln_lk.backward()

            # call user's callback
            with torch.no_grad():
                epoch_callback(epoch, self._regression, neg_ln_lk.item())

            self._optimizer.step()
            if self._lr_sched is not None:
                self._lr_sched.step()
            if self._projector is not None:
                self._projector.Project()

            # debug
            epoch_iter.set_description(f"Loss: {neg_ln_lk.item()}")

            # break condition
            if self._paramsNormGradient() < self._grad_tol:
                return






