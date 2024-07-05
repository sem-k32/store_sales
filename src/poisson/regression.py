import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from typing import List
from functools import reduce


class PoissonRegr:
    def __init__(self, lambda_computer: nn.Module, data_list: List[pd.DataFrame] = None) -> None:
        self._lambd_func = lambda_computer
        self._data_list = data_list

    def setData(self, data_list: List[pd.DataFrame]) -> None:
        self._data_list = data_list

    def lnLiklyhood(self) -> torch.Tensor:
        if self._data_list is None:
            raise ValueError("No data set to compute liklyhood")

        output = 0

        # compute normalizing constant
        num_data_obj = reduce(lambda num_obj, data: num_obj + data.shape[0], self._data_list, 0)
        
        for data in self._data_list:
            num_events = torch.from_numpy(
                data["sales"].values.astype(np.float64)
            )
            lambdas: torch.Tensor = self._lambd_func(data)

            # normalized LH
            output += torch.sum((num_events * torch.log(lambdas) - lambdas))

        return output
    
    def getLambdas(self, period_info: pd.DataFrame) -> torch.Tensor:
        with torch.no_grad():
            return self._lambd_func(period_info)