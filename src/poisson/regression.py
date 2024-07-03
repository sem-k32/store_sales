import torch
import torch.nn as nn
import pandas as pd

from typing import List


class PoissonRegr:
    def __init__(self, lambda_computer: nn.Module, data_list: List[pd.DataFrame] = None) -> None:
        self._lambd_func = lambda_computer
        self._data_list = data_list

    def setData(self, data_list: List[pd.DataFrame]) -> None:
        self._data_list = data_list

    def lnLiklyhood(self) -> torch.Tensor:
        output = 0
        
        for data in self._data_list:
            num_events = torch.from_numpy(
                data["sales"].values()
            )
            lambdas: torch.Tensor = self._lambd_func(data)

            output = output + (num_events * torch.log(lambdas) - lambdas).sum()

        return output
    
    def getLambdas(self, period_info: pd.DataFrame) -> torch.Tensor:
        return self._lambd_func(period_info)