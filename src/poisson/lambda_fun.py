""" Poisson lambda parameter approximators
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import calendar
from abc import ABC, abstractmethod


# Linear month-wise models

class lambdaLinearSpline(ABC, nn.Module):
    def __init__(self, promotion_init: float, coefs_init: list[torch.DoubleTensor]):
        """The model is linear, requires time transformation into feature vec. Assumes year-periodicity. 
            Using month-wise approximation

        Model's order is defined by initial values for params.

        Args:
            promotion_init (float): initial value for the promotion param
            coefs_init (list[torch.DoubleTensor]): intial coeffs for every month
        """
        super().__init__()

        # create model's params
        NUM_MONTHS = 12
        self._month_coefs = nn.ParameterList()
        for month in range(NUM_MONTHS):
            self._month_coefs.append(nn.Parameter(coefs_init[month], requires_grad=True))

        # onpromotion constant
        self._promotion_c = nn.Parameter(torch.DoubleTensor([promotion_init]), requires_grad=True)

        # compute number of days in each month for leap year
        # this will be enough for lambdas computations at any time
        self._days_per_month = [calendar.monthrange(2024, month)[1] for month in range(1, 12 + 1)]

    def forward(self, series: pd.DataFrame) -> torch.Tensor:
        num_time_points = series.shape[0]
        output_lambdas = torch.empty(num_time_points, dtype=torch.float64)

        for i in range(num_time_points):
            cur_date = pd.to_datetime(series.iloc[i]["date"])
            num_promotion = torch.DoubleTensor([series.iloc[i]["onpromotion"]])

            # new year law
            if cur_date.month == 1 and cur_date.day == 1:
                output_lambdas[i] = 0
                continue

            feature_vec = self._getFeatureVec(cur_date)
            # days/months for lambdas start with zero, onpromotion bias
            output_lambdas[i] = torch.dot(feature_vec, self._month_coefs[cur_date.month - 1]) + \
                                        self._promotion_c * num_promotion

        return output_lambdas

    @abstractmethod
    def _getFeatureVec(self, time: pd.Timestamp) -> torch.Tensor:
        ...


class lambdaPolySpline(lambdaLinearSpline):
    def __init__(self, promotion_init: float, coefs_init: list[torch.DoubleTensor]):
        """linear model using polynomial features. Polys' coefs are from low order to high
        """
        super().__init__(promotion_init, coefs_init)

        self._model_ord = coefs_init[0].shape[0] - 1

    def _getFeatureVec(self, time: pd.Timestamp) -> torch.Tensor:
        # time is normed relative to number of days in a month
        normed_time = time.day / self._days_per_month[time.month - 1]
        return torch.DoubleTensor([normed_time ** j for j in range(0, self._model_ord + 1)])
    

class lambdaCosineSpline(lambdaLinearSpline):
    def __init__(self, promotion_init: float, coefs_init: list[torch.DoubleTensor]):
        """linear model using harmonical features. Coefs goes: free coefs, sin coefs, cos coefs
        """
        super().__init__(promotion_init, coefs_init)

        self._model_ord = coefs_init[0].shape[0] - 1

    def _getFeatureVec(self, time: pd.Timestamp) -> torch.Tensor:
        # time is normed relative to number of days in a month
        normed_time = time.day / self._days_per_month[time.month - 1]
        fourier_freq = 2 * np.pi

        # cos_coefs includes free ratio
        cos_coefs = [np.cos(fourier_freq * j * normed_time) ** 2 for j in range(1, self._model_ord)]

        return torch.DoubleTensor(cos_coefs)
    

# Abs variation of linear models

class lambdaAbsPolySpline(lambdaPolySpline):
    def __init__(self, promotion_init: float, coefs_init: list[torch.DoubleTensor]) -> None:
        """ the same model as lambdaPolySpline, but envelopes lambda function in abs().
                Does not require positivity of the coeffs
        """
        super().__init__(promotion_init, coefs_init)

    def forward(self, series: pd.DataFrame) -> torch.Tensor:
        return torch.abs(super().forward(series))


# Linear models

class lambdaLinear(ABC, nn.Module):
    def __init__(self, promotion_init: float, coefs_init: torch.DoubleTensor):
        """The model is linear, requires time transformation into feature vec. Assumes year-periodicity. 

        Model's order is defined by initial values for params.

        Args:
            promotion_init (float): initial value for the promotion param
            coefs_init (list[torch.DoubleTensor]): intial coeffs
        """
        super().__init__()

        self._model_ord = coefs_init.shape[0] - 1

        # create model's params
        self._coefs = nn.Parameter(coefs_init, requires_grad=True)

        # onpromotion constant
        self._promotion_c = nn.Parameter(torch.DoubleTensor([promotion_init]), requires_grad=True)

    def forward(self, series: pd.DataFrame) -> torch.Tensor:
        num_time_points = series.shape[0]
        output_lambdas = torch.empty(num_time_points, dtype=torch.float64)

        for i in range(num_time_points):
            cur_date = pd.to_datetime(series.iloc[i]["date"])
            num_promotion = torch.DoubleTensor([series.iloc[i]["onpromotion"]])

            # new year law
            if cur_date.month == 1 and cur_date.day == 1:
                output_lambdas[i] = 0
                continue

            feature_vec = self._getFeatureVec(cur_date)
            # days/months for lambdas start with zero, onpromotion bias
            output_lambdas[i] = torch.dot(feature_vec, self._coefs) + self._promotion_c * num_promotion
            pass

        return output_lambdas
    

class lambdaPoly(lambdaLinear):
    def __init__(self, promotion_init: float, coefs_init: torch.DoubleTensor):
        """linear model using polynomial features. Polys' coefs are from low order to high
        """
        super().__init__(promotion_init, coefs_init)

        self._model_ord = coefs_init[0].shape[0] - 1

    def _getFeatureVec(self, time: pd.Timestamp) -> torch.Tensor:
        # time is normed relative to number of days in a year
        normed_time = time.dayofyear / (365 + time.is_leap_year)
        return torch.DoubleTensor([normed_time ** j for j in range(0, self._model_ord + 1)])
    

class lambdaCosine(lambdaLinear):
    def __init__(self, promotion_init: float, coefs_init: torch.DoubleTensor):
        """linear model using harmonical features. Coefs goes: free coefs, cos coefs
        """
        super().__init__(promotion_init, coefs_init)

        self._model_ord = coefs_init.shape[0] - 1

    def _getFeatureVec(self, time: pd.Timestamp) -> torch.Tensor:
        # time is normed relative to number of days in a year
        normed_time = time.dayofyear / (365 + time.is_leap_year)
        fourier_freq = 2 * np.pi

        # cos coefs includes free ratio
        cos_coefs = [np.cos(fourier_freq * j * normed_time) ** 2 for j in range(0, self._model_ord + 1)]

        return torch.DoubleTensor(cos_coefs)
    

class lambdaFouirer(lambdaLinear):
    def __init__(self, promotion_init: float, coefs_init: torch.DoubleTensor):
        """linear model using harmonical features. Coefs goes: free coefs, cos coefs, sin coefs
        """
        super().__init__(promotion_init, coefs_init)

        self._model_ord = (coefs_init.shape[0] - 1) // 2

    def _getFeatureVec(self, time: pd.Timestamp) -> torch.Tensor:
        # time is normed relative to number of days in a year
        normed_time = time.dayofyear / (365 + time.is_leap_year)
        fourier_freq = 2 * np.pi

        # cos coefs includes free ratio
        cos_coefs = [np.cos(fourier_freq * j * normed_time) for j in range(0, self._model_ord + 1)]
        sin_coefs = [np.sin(fourier_freq * j * normed_time) for j in range(1, self._model_ord + 1)]

        return torch.DoubleTensor(cos_coefs + sin_coefs)
    

# Models using abs()

class lambdaAbsFouirer(lambdaFouirer):
    def __init__(self, promotion_init: float, coefs_init: torch.DoubleTensor):
        """ the same model as lambdaFouirer, but envelopes lambda function in abs().
                Does not require positivity of the coeffs
        """
        super().__init__(promotion_init, coefs_init)

    def forward(self, series: pd.DataFrame) -> torch.Tensor:
        return torch.abs(super().forward(series))