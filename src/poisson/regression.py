import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from typing import List
from functools import reduce


class PoissonRegr:
    def __init__(self, lambda_func: nn.Module) -> None:
        """the class gives a liklyhood function to poisson data model and gives a lambda parameter for given 
            time and the time's additional info

        Args:
            lambda_func (nn.Module): approximator of the poisson's lambda parameter
        """
        self._lambd_func = lambda_func

    def negLnLiklyhood(self, time_info: pd.DataFrame) -> torch.Tensor:
        """compute data's negative log Liklyhood for given time period

        Args:
            time_info (pd.DataFrame): sales in a given time + additional info
        """
        if time_info.size == 0:
            raise ValueError("Given period info is empty")

        num_events = torch.from_numpy(
            time_info["sales"].values.astype(np.float64)
        )
        lambdas: torch.Tensor = self._lambd_func(time_info)

        return torch.sum((lambdas - num_events * torch.log(lambdas)).nan_to_num(0, 0, 0))
    
    def getLambdas(self, time_info: pd.DataFrame) -> torch.Tensor:
        """ compute lambda params for given time period. Useful for forecasting

        Args:
            time_info (pd.DataFrame): time period with additional info
        """
        if time_info.size == 0:
            raise ValueError("Given period info is empty")
        
        with torch.no_grad():
            return self._lambd_func(time_info)