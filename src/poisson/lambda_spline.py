import torch
import torch.nn as nn
import pandas as pd

import calendar

class lambdaSpline(nn.Module):
    def __init__(self, order: int, smoothness: int) -> None:
        """
        Args:
            order (int): for the polynoms to use
            smoothness (int): how smooth should be polys in conjecture points

        Raises:
            ValueError: if poly's order is less then smoothness parameter
        """
        self._poly_ord = order
        self._smooth = smoothness
        if smoothness >= order:
            raise ValueError("Smoothness should be less than poly's order")

        # define polys coeffs
        NUM_MONTHS = 12
        self._polys = nn.ParameterList()
        for month in range(NUM_MONTHS):
            # poly's coeffs are from highest to lowest
            self._polys.append(nn.Parameter(torch.rand(order), requires_grad=True))

        # onpromotion constant, positive
        self._promotion_c = nn.Parameter(torch.ones(1), requires_grad=True)

        # compute number of days in each month for non-leap year
        self._days_per_month = [calendar.monthrange(2023, month)[1] for month in range(1, 12 + 1)]

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
            computed_lambdas[month] = torch.dot(
                torch.vander(torch.arange(0, self._days_per_month[month]), self._poly_ord),
                self._polys[month]
            )

        # iterate over months and paste relevant lambdas
        output_lambdas = torch.empty(0)
        dates = series["date"]
        cur_month = dates[0]

        while True: 
            cur_month_days = dates.loc[(dates.dt.year == cur_month.dt.year) & (dates.dt.month == cur_month.dt.month)].dt.day
            if cur_month_days.empty:
                break
            else:
                cur_month_days = (cur_month_days.values() - 1).tolist()
                output_lambdas = torch.concat(
                    [output_lambdas, computed_lambdas[cur_month.month][cur_month_days]],
                    dim=0
                )

                cur_month = cur_month + pd.DateOffset(months=1)

        # add onpromotion bias
        is_on_promotion = (series["onpromotion"] > 0).values()
        output_lambdas += torch.from_numpy(is_on_promotion) * self._promotion_c

        return output_lambdas
    
