""" Data preprocessing utilites
"""

import numpy as np
import pandas as pd


class zerosRemovePreprocess:
    def __init__(self, max_conseq_allowed: int) -> None:
        """remove consequtive zero sales points from data

        Args:
            max_conseq_allowed (int, optional): allowed length of consequtive zeros. Defaults to 5.
        """
        self._max_conseq_allowed = max_conseq_allowed

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        output = []

        sales_data = data["sales"].values

        left = 0
        num_zeros = 0
        i = 0
        while i < sales_data.shape[0]:
            cur_sale = sales_data[i]
            if cur_sale == 0.0:
                num_zeros += 1
            elif num_zeros > self._max_conseq_allowed:
                output.append(data.iloc[left: i - num_zeros])
                left = i
                num_zeros = 0
            else:
                num_zeros = 0

            i += 1
        
        # check last zeros overflow or add last chunk of data
        if num_zeros > self._max_conseq_allowed:
            output.append(data.iloc[left: i - num_zeros])
        else:
            output.append(data.iloc[left: i])

        return pd.concat(output)


class earthquaquePreprocess:
    def __init__(self) -> None:
        """remove dates accosiated with earthqueque in Ecvador
        """
        self._quaque_start = pd.to_datetime("04-16-2016");
        self._quaque_end = (pd.to_datetime("04-16-2016").to_period(freq="D") + 14).to_timestamp();

    def __call__(self, data: pd.DataFrame) -> pd.DateOffset:
        return data.drop(
            data[(data["date"] >= self._quaque_start) & (data["date"] >= self._quaque_end)].index
        )


class percentilePreprocess:
    def __init__(self, percentile: float) -> None:
        """removes all sales for given data which exceeds empirical percentile
        """
        self._precentile = percentile

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        sales_percent_val = data["sales"].quantile(self._precentile)

        return data.drop(
            data[data["sales"] >= sales_percent_val].index
        )
