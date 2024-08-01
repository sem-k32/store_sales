import pandas as pd

from src.data.preprocessing import *

import pathlib
import os
from yaml import full_load


if __name__ == "__main__":
    # load cwd and params
    cwd = pathlib.Path(os.environ["WORKSPACE_DIR"])
    with open("params.yaml", "r") as f:
        params_dict = full_load(f)
    
    # create data dir
    exp_data_dir = pathlib.Path(params_dict["data_dir"])
    exp_data_dir.mkdir(exist_ok=True)

    # load all neccessary data
    train_data_all = pd.read_csv(cwd / "data/train.csv")
    test_data_all = pd.read_csv(cwd / "data/test.csv")
    # transform date column to date format
    train_data_all.loc[:, "date"] = pd.to_datetime(train_data_all["date"])
    test_data_all.loc[:, "date"] = pd.to_datetime(test_data_all["date"])

    # make train/validate/test .csv files

    train_data = train_data_all.loc[
        (train_data_all["store_nbr"] == params_dict["store_nbr"]) &
        (train_data_all["family"] == params_dict["family"]) &
        (train_data_all["date"] < pd.to_datetime(params_dict["validate_data_date"]))
    ]
    # preprocessing of train data
    train_data = percentilePreprocess(0.98)(
        earthquaquePreprocess()(
            zerosRemovePreprocess(5)(train_data)
        )
    )
    train_data.to_csv(exp_data_dir / "train.csv")

    validate_data = train_data_all.loc[
        (train_data_all["store_nbr"] == params_dict["store_nbr"]) &
        (train_data_all["family"] == params_dict["family"]) &
        (train_data_all["date"] >= pd.to_datetime(params_dict["validate_data_date"]))
    ]
    validate_data.to_csv(exp_data_dir / "validate.csv")

    test_data = test_data_all.loc[
        (test_data_all["store_nbr"] == params_dict["store_nbr"]) &
        (test_data_all["family"] == params_dict["family"])
    ]
    test_data.to_csv(exp_data_dir / "test.csv")
