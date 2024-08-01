""" evaluate model's performance on train and validate data
"""
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from matplotlib import pyplot as plt
from dvclive import Live

import pathlib
from yaml import full_load

# src is already in PYTHONPATH
from src.poisson.lambda_fun import lambdaAbsPolySpline
from src.poisson.regression import PoissonRegr
from src.poisson.trainer import Trainer


def promotionInitializer() -> float:
    return 1.0

def polyInitializer(order: int) -> list[torch.Tensor]:
    NUM_MONTHS = 12
    return [
        1e-1 * torch.ones(order + 1, dtype=torch.float64)
        for _ in range(NUM_MONTHS)
    ]

def paramsNormGradient(lambda_func) -> float:
    with torch.no_grad():
        grad_norm = 0
        for param in lambda_func.parameters():
            grad_norm += torch.sum(param.grad ** 2)

        return torch.sqrt(grad_norm).item()


if __name__ == "__main__":
    # load params
    with open("params.yaml", "r") as f:
        params_dict = full_load(f)

    # load data
    train_data = pd.read_csv("exp_data/train.csv")
    validate_data = pd.read_csv("exp_data/validate.csv")

    # init model
    lambda_func = lambdaAbsPolySpline(
        promotionInitializer(),
        polyInitializer(params_dict["poly_ord"])
    )
    regressor = PoissonRegr(lambda_func)

    # optimizer
    optimizer = optim.SGD(lambda_func.parameters(), lr=params_dict["lr"],
                           momentum=params_dict["momentum"], nesterov=True)
    # constant lr
    lr_sched = LambdaLR(optimizer, lambda epoch: params_dict["lr"])
    # model's params projector
    projector = None
    trainer = Trainer(
        regressor,
        train_data,
        optimizer,
        lr_sched,
        projector,
        params_dict["max_epochs"],
        params_dict["grad_tol"]
    )

    # create results dir
    results_dir = pathlib.Path(params_dict["eval_results_dir"])
    results_dir.mkdir(exist_ok=True)
    # logger
    logger = Live(results_dir)
    # train callback (is called under torch.no_grad() inside trainer)
    def train_callback(epoch: int, regressor: PoissonRegr, neg_ln_lk: float) -> None:
        with torch.no_grad():
            logger.log_metric("Train/neg_ln_liklyhood", neg_ln_lk)
            logger.log_metric("Train/grad_norm", paramsNormGradient(regressor._lambd_func))
            logger.log_metric("Test/neg_ln_liklyhood", regressor.negLnLiklyhood(validate_data).item())
            # Kaggle prediction metric
            observations = torch.from_numpy(validate_data["sales"].values).to(torch.float64)
            prediction = regressor.getLambdas(validate_data)
            kaggle_metric = torch.sqrt(torch.mean((torch.log(1 + prediction) - torch.log(1 + observations)) ** 2))
            logger.log_metric("Test/kaggle_metric", kaggle_metric.item())

            if epoch % 5 == 0:
                # vizualize solution
                observations = validate_data["sales"].values
                prediction = regressor.getLambdas(validate_data).numpy()
                pred_dates = validate_data["date"].values.astype("datetime64")

                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(pred_dates, observations, label="observe")
                ax.plot(pred_dates, prediction, label="predict", color="red")
                ax.fill_between(pred_dates, prediction - np.sqrt(prediction), prediction + np.sqrt(prediction), 
                                color="orange", alpha=0.6, label="predict_dispersion"
                )

                ax.grid(True)
                ax.legend()
                ax.set_xlabel("t")
                ax.set_ylabel("Sales")

                logger.log_image(f"prediction_{epoch}.png", fig)

                # print model's coefs
                model_state_dict = lambda_func.state_dict()
                for key, param in model_state_dict.items():
                    print(key, param)

            logger.next_step()

    # train model
    trainer.Train(train_callback)

    # print model's coefs
    model_state_dict = lambda_func.state_dict()
    for key, param in model_state_dict.items():
        print(key, param)

    # vizualize validation solution

    observations = validate_data["sales"].values
    prediction = regressor.getLambdas(validate_data).numpy()
    pred_dates = validate_data["date"].values.astype("datetime64")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(pred_dates, observations, label="observe")
    ax.plot(pred_dates, prediction, label="predict", color="red")
    ax.fill_between(pred_dates, prediction - np.sqrt(prediction), prediction + np.sqrt(prediction), 
                    color="orange", alpha=0.6, label="predict_variance"
    )

    ax.grid(True)
    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("Sales")

    logger.log_image("prediction_final.png", fig)

    # vizualize train solution

    observations = train_data["sales"].values
    prediction = regressor.getLambdas(train_data).numpy()
    pred_dates = train_data["date"].values.astype("datetime64")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(pred_dates, observations, label="observe")
    ax.plot(pred_dates, prediction, label="predict", color="red")
    ax.fill_between(pred_dates, prediction - np.sqrt(prediction), prediction + np.sqrt(prediction), 
                    color="orange", alpha=0.6, label="predict_variance"
    )

    ax.grid(True)
    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("Sales")

    logger.log_image("train_final.png", fig)

    # end evaluation stage
    logger.end()

