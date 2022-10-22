# Copyright (c) 2021, Kamilė Stankevičiūtė
# Licensed under the BSD 3-clause license

import gc
import pickle

import torch

from models.bjrnn import RNN_uncertainty_wrapper
from models.cfrnn import CFRNN, AdaptiveCFRNN
from models.dprnn import DPRNN
from models.qrnn import QRNN
from models.rnn import RNN
from utils.data_processing_synthetic import (
    DEFAULT_PARAMETERS,
    EXPERIMENT_MODES,
    get_raw_sequences,
    get_synthetic_dataset,
)
from utils.performance import evaluate_cfrnn_performance, evaluate_performance

BASELINES = {"CFRNN": CFRNN, "AdaptiveCFRNN": AdaptiveCFRNN, "BJRNN": None, "DPRNN": DPRNN, "QRNN": QRNN}

CONFORMAL_BASELINES = ["CFRNN", "AdaptiveCFRNN"]

DEFAULT_SYNTHETIC_TRAINING_PARAMETERS = {
    "input_size": 1,  # RNN parameters
    "epochs": 10,
    "normaliser_epochs": 1000,
    "n_steps": 500,
    "batch_size": 100,
    "embedding_size": 20,
    "max_steps": 10,
    "horizon": 5,
    "coverage": 0.9,
    "lr": 0.01,
    "rnn_mode": "LSTM",
    "beta": 1,
}


def get_max_steps(train_dataset, test_dataset):
    return max(max(train_dataset[2]), max(test_dataset[2]))


def get_model_path(experiment, rnn_mode, mode, seed, dynamic_sequence_lengths, horizon, baseline=None):
    return "saved_models/{}-{}-{}-{}-{}{}{}.pt".format(
        experiment,
        "aux" if baseline is None else baseline,
        rnn_mode,
        mode,
        seed,
        ("-dynamic" if dynamic_sequence_lengths else ""),
        (
            "-horizon{}".format(horizon)
            if horizon is not None and horizon != DEFAULT_SYNTHETIC_TRAINING_PARAMETERS["horizon"]
            else ""
        ),
    )


def get_results_path(experiment, baseline, seed, dynamic_sequence_lengths, horizon):
    return "saved_results/{}-{}-{}{}{}.pkl".format(
        experiment,
        baseline,
        seed,
        ("-dynamic" if dynamic_sequence_lengths else ""),
        (
            "-horizon{}".format(horizon)
            if horizon is not None and horizon != DEFAULT_SYNTHETIC_TRAINING_PARAMETERS["horizon"]
            else ""
        ),
    )


def run_synthetic_experiments(
    experiment,
    baseline,
    retrain_auxiliary=False,
    recompute_dataset=False,
    params=None,
    dynamic_sequence_lengths=False,
    n_train=None,
    horizon=None,
    beta=None,
    correct_conformal=True,
    save_model=False,
    save_results=True,
    rnn_mode=None,
    seed=0,
):
    """
    Runs an experiment for a synthetic dataset.

    Args:
        experiment: type of experiment ('time_dependent', 'static', 'periodic', 'sample_complexity')
        baseline: the model to be trained ('BJRNN', 'DPRNN', 'QRNN', 'CFRNN', 'AdaptiveCFRNN')
        retrain_auxiliary: whether to retrain the AuxiliaryForecaster of the CFRNN models
        recompute_dataset: whether to generate the dataset from scratch
        params: dictionary of training parameters
        dynamic_sequence_lengths: whether to use datasets where sequences have different randomly sampled lengths
        n_train: number of training examples
        horizon: forecasting horizon
        beta: (in AdaptiveCFRNN) hyperparameter to dampen the importance of the correction factor
        correct_conformal: whether to use Bonferroni-corrected calibration scores
        save_model: whether to save the model in the `./saved_models/` directory
        save_results: whether to save the results in `./saved_results/`
        rnn_mode: (in CFRNN) the type of RNN of the underlying forecaster (RNN/LSTM/GRU)
        seed: random seed

    Returns:
        a dictionary of result metrics
    """

    assert baseline in BASELINES.keys(), "Invalid baseline"
    assert experiment in EXPERIMENT_MODES.keys(), "Invalid experiment"

    baseline_results = []

    torch.manual_seed(seed)

    raw_sequence_datasets = get_raw_sequences(
        experiment=experiment,
        n_train=n_train,
        dynamic_sequence_lengths=dynamic_sequence_lengths,
        horizon=horizon,
        seed=seed,
        recompute_dataset=recompute_dataset,
    )
    print("Training {}".format(baseline))

    for i, raw_sequence_dataset in enumerate(raw_sequence_datasets):
        print("Training dataset {}".format(i))

        if params is None:
            params = DEFAULT_SYNTHETIC_TRAINING_PARAMETERS.copy()

        if rnn_mode is not None:
            params["rnn_mode"] = rnn_mode

        if beta is not None:
            params["beta"] = beta

        if horizon is not None:
            params["horizon"] = horizon

        if not retrain_auxiliary:
            auxiliary_forecaster_path = get_model_path(
                experiment, params["rnn_mode"], EXPERIMENT_MODES[experiment][i], seed, dynamic_sequence_lengths, horizon
            )
        else:
            auxiliary_forecaster_path = None

        params["output_size"] = horizon if horizon else DEFAULT_PARAMETERS["horizon"]

        if baseline in CONFORMAL_BASELINES:
            params["epochs"] = 1000

            train_dataset, calibration_dataset, test_dataset = get_synthetic_dataset(
                raw_sequence_dataset, conformal=True, seed=seed
            )

            # print(train_dataset[0])
            # print output is two tensors of shape [15,1](training part of a sequence) and [5,1](target part of a sequence) respectively,
            # and 15(which is the sequence_length of training part)

            model = BASELINES[baseline](
                embedding_size=params["embedding_size"],
                horizon=params["horizon"],
                error_rate=1 - params["coverage"],
                rnn_mode=params["rnn_mode"],
                auxiliary_forecaster_path=auxiliary_forecaster_path,
                beta=params["beta"],
            )
            model.fit(
                train_dataset,
                calibration_dataset,
                epochs=params["epochs"],
                lr=params["lr"],
                batch_size=params["batch_size"],
                normaliser_epochs=params["normaliser_epochs"],
            )

            result = evaluate_cfrnn_performance(model, test_dataset, correct_conformal)

        else:
            train_dataset, test_dataset = get_synthetic_dataset(raw_sequence_dataset, conformal=False, seed=seed)

            if dynamic_sequence_lengths or horizon is None:
                params["max_steps"] = get_max_steps(train_dataset, test_dataset)

            if baseline == "BJRNN":
                params["epochs"] = 1000
                RNN_model = RNN(**params)
                RNN_model.fit(train_dataset[0], train_dataset[1])
                model = RNN_uncertainty_wrapper(RNN_model, rnn_mode="RNN")
            else:
                model = BASELINES[baseline](**params)
                model.fit(train_dataset[0], train_dataset[1])

            result = evaluate_performance(model, test_dataset[0], test_dataset[1], coverage=params["coverage"])

        baseline_results.append(result)

        if save_model:
            torch.save(
                model,
                get_model_path(
                    experiment,
                    model.rnn_mode,
                    EXPERIMENT_MODES[experiment][i],
                    seed,
                    dynamic_sequence_lengths,
                    horizon,
                    baseline,
                ),
            )

        del model
        gc.collect()

    if save_results:
        with open(get_results_path(experiment, baseline, seed, dynamic_sequence_lengths, horizon), "wb") as f:
            pickle.dump(baseline_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return baseline_results


def load_synthetic_results(experiment, baseline, seed=0, horizon=None, dynamic_sequence_lengths=False):
    path = get_results_path(experiment, baseline, seed, dynamic_sequence_lengths, horizon)
    with open(path, "rb") as f:
        baseline_results = pickle.load(f)

    return baseline_results
