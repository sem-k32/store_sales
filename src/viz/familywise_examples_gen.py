""" script saves several/all time series examples for each family
"""
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import pathlib
from typing import Optional


def getHolidayDates(store: int, holidays: pd.DataFrame, stores_info: pd.DataFrame):
    store_info = stores_info[stores_info["store_nbr"] == store]
    store_city = store_info["city"].values[0]
    store_state = store_info["state"].values[0]

    relevant_holidays = holidays.loc[
        ((holidays["locale_name"] == store_city) | (holidays["locale_name"] == store_state) | 
         (holidays["locale"] == "National")),
         "date"
    ]

    return relevant_holidays

def genFamilywiseExamples(
        examples_per_fam: Optional[int]
) -> None:
    pathlib.Path("img/examples").mkdir(exist_ok=True)

    data = pd.read_csv("data/train.csv")
    holidays = pd.read_csv("data/holidays_events.csv")
    stores_info = pd.read_csv("data/stores.csv")

    families = data["family"].unique()

    for family in families:
        if family.find("/") != -1:
            save_dir = f"img/examples/BAKERY_BREAD"
        else:
            save_dir = f"img/examples/{family}"

        print(f"Processing {family}")

        pathlib.Path(save_dir).mkdir(exist_ok=True)

        possible_stores: pd.Series = data.loc[data["family"] == family, "store_nbr"].unique()
        # sample random/all stores
        if examples_per_fam is None:
            chosen_stores = possible_stores
        else:
            chosen_stores = np.random.choice(possible_stores, examples_per_fam, replace=False)

        for store in chosen_stores:
            series = data.loc[
                (data["family"] == family) & (data["store_nbr"] == store),
                ["date", "sales", "onpromotion"]
            ]
            on_promotion_points = series.loc[series["onpromotion"] > 0, ["date", "sales"]]
            holiday_dates = getHolidayDates(store, holidays, stores_info)
            holiday_sales = series.loc[series["date"].isin(holiday_dates), ["date", "sales"]]

            fig, ax = plt.subplots(figsize=(15, 8))
            # plot series
            ax.plot(series["date"].astype('datetime64[s]'), series["sales"], marker=".")
            # plot onpromotion sales
            ax.scatter(on_promotion_points["date"].astype('datetime64[s]'), on_promotion_points["sales"],
                        color="green", label="prom")
            # plot holiday sales
            ax.scatter(holiday_sales["date"].astype('datetime64[s]'), holiday_sales["sales"],
                        color="red", label="holiday")

            ax.set_xlabel("t")
            ax.set_ylabel("Sales")
            ax.set_title(f"{family}, store {store}")
            ax.grid(True)
            ax.legend()

            fig.savefig(f"{save_dir}/store_{store}.png", format="png")
            plt.close(fig)


if __name__ == "__main__":
    genFamilywiseExamples(examples_per_fam=None)


