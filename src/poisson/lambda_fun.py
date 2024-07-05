import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import calendar

class lambdaSpline(nn.Module):
    def __init__(self, order: int) -> None:
        """
        Args:
            order (int): for the polynoms to use

        Raises:
            ValueError: if poly's order is less then smoothness parameter
        """
        super().__init__()

        self._poly_ord = order

        # define polys coeffs
        NUM_MONTHS = 12
        self._polys = nn.ParameterList()
        for month in range(NUM_MONTHS):
            # poly's coeffs are from highest to lowest
            self._polys.append(nn.Parameter(2 + torch.rand(order + 1, dtype=torch.float64), requires_grad=True))

        # onpromotion constant, positive
        self._promotion_c = nn.Parameter(torch.ones(1, dtype=torch.float64), requires_grad=True)

        # compute number of days in each month for leap year
        # this will be enough for lambdas computations
        self._days_per_month = [calendar.monthrange(2024, month)[1] for month in range(1, 12 + 1)]

    def forward(self, series: pd.DataFrame) -> torch.Tensor:
        """

        Args:
            series (pd.DataFrame): timestamps and additional fields vital for lambda computing;
                                    timestamps should be strictly consequtive

        Returns:
            torch.Tensor: lambdas for given timestamps
        """
        # compute lambdas for presented months
        computed_lambdas = {}
        for month in series["date"].dt.month.unique() - 1:
            month = int(month)

            # we norm timeline into [0, 1]
            computed_lambdas[month] = torch.abs(torch.matmul(
                torch.flip(
                    torch.vander(
                        torch.arange(0, self._days_per_month[month], dtype=torch.float64) / (self._days_per_month[month] - 1), 
                        self._poly_ord + 1
                    ), 
                    dims=[0]
                ),
                self._polys[month]
            ))

        # iterate over months and paste relevant lambdas
        lambda_list = []
        dates = series["date"]
        cur_month = pd.to_datetime(dates.values[0]).replace(day=1)

        while True: 
            cur_month_days = dates.loc[(dates.dt.year == cur_month.year) & (dates.dt.month == cur_month.month)].dt.day
            if cur_month_days.empty:
                break
            else:
                # days/months for lambdas start with zero
                cur_month_days = (cur_month_days.values - 1).tolist()
                lambda_list.append(computed_lambdas[cur_month.month - 1][cur_month_days])

                cur_month = cur_month + pd.DateOffset(months=1)

        # add onpromotion bias
        output_lambdas = torch.concat(lambda_list)
        is_on_promotion = (series["onpromotion"] > 0).values.tolist()
        output_lambdas[is_on_promotion] += self._promotion_c

        return output_lambdas
    