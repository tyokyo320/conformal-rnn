# Copyright (c) 2022, tyokyo320
# Licensed under the BSD 3-clause license

import pickle
import torch
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.preprocessing import StandardScaler


class ElectricityDataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, sequence_lengths) -> None:
        super(ElectricityDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.sequence_lengths = sequence_lengths
        
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, key: int):
        return self.X[key], self.Y[key], self.sequence_lengths[key]


def get_raw_electricity_data(cached=True) -> np.ndarray:
    areas = [
        'CAPITL',
        'CENTRL',
        'DUNWOD',
        'GENESE',
        'HUDVL',
        'LONGIL',
        'MHKVL',
        'MILLWD',
        'N_Y_C_',
        'NORTH',
        'WEST',
    ]

    if cached:
        with open("data/nyiso.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        df = pd.read_csv("data/nyiso_5min.csv")
        for area in areas:
            # [1 day -> 1 hour] select 300 data (length=288 + horizon=12)
            # dataset.append(df[area].to_numpy()[-400:-100])
            # [1 week -> 1 hour] select 2028 data (length=2016 + horizon=12)
            dataset.append(df[area].to_numpy()[0:2028])
        dataset = np.array(dataset)
        # print(f'dataset = {dataset}, dataset length = {len(dataset)}')
        with open("data/nyiso.pkl", "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


def get_electricity_splits(length=2016, horizon=12, conformal=True, n_train=6, n_calibration=4, n_test=1, cached=True, seed=None) -> ElectricityDataset:
    if seed is None:
        seed = 0
    else:
        cached = False

    if cached:
        if conformal:
            with open("processed_data/nyiso_conformal.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
        else:
            with open("processed_data/nyiso_raw.pkl", "rb") as f:
                train_dataset, calibration_dataset, test_dataset = pickle.load(f)
    else:
        # raw_data shape(380, 150)
        np.set_printoptions(threshold=np.inf)
        raw_data = get_raw_electricity_data(cached=cached)
        # print(f'dataset = {raw_data}, raw_data length = {len(raw_data)}')
        # X shape(380, 100)
        # Y shape(380, 50)
        X = raw_data[:, :length]
        Y = raw_data[:, length : length + horizon]
        # print(f'X = {X}, X length = {len(X)}')
        # print(f'Y = {Y}, Y length = {len(Y)}')

        perm = np.random.RandomState(seed=seed).permutation(n_train + n_calibration + n_test)
        # print(f'perm = {perm}, perm length = {len(perm)}')
        train_idx = perm[:n_train]
        calibration_idx = perm[n_train : n_train + n_calibration]
        train_calibration_idx = perm[: n_train + n_calibration]
        test_idx = perm[n_train + n_calibration :]
        # print(f'train_idx = {train_idx}, train_idx length = {len(train_idx)}')
        # print(f'calibration_idx = {calibration_idx}, calibration_idx length = {len(calibration_idx)}')
        # print(f'train_calibration_idx = {train_calibration_idx}, train_calibration_idx length = {len(train_calibration_idx)}')
        # print(f'test_idx = {test_idx}, test_idx length = {len(test_idx)}')

        if conformal:
            X_train = X[train_idx]
            X_calibration = X[calibration_idx]
            X_test = X[test_idx]
            # print(f'X_train = {X_train}, X_train length = {len(X_train)}')
            # print(f'X_calibration = {X_calibration}, X_calibration length = {len(X_calibration)}')
            # print(f'X_test = {X_test}, X_test length = {len(X_test)}')

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_calibration_scaled = scaler.transform(X_calibration)

            train_dataset = ElectricityDataset(
                torch.FloatTensor(X_train_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[train_idx]).reshape(-1, horizon, 1),
                torch.ones(len(train_idx), dtype=torch.int) * length,
            )

            calibration_dataset = ElectricityDataset(
                torch.FloatTensor(X_calibration_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[calibration_idx]).reshape(-1, horizon, 1),
                torch.ones(len(calibration_idx)) * length,
            )

            test_dataset = ElectricityDataset(
                torch.FloatTensor(X_test_scaled).reshape(-1, length, 1),
                torch.FloatTensor(Y[test_idx]).reshape(-1, horizon, 1),
                torch.ones(len(X_test_scaled), dtype=torch.int) * length,
            )

            with open("processed_data/nyiso_conformal.pkl", "wb") as f:
                pickle.dump((train_dataset, calibration_dataset, test_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            X_train = X[train_calibration_idx]
            X_test = X[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            train_dataset = X_train_scaled, Y[train_calibration_idx]
            calibration_dataset = None
            test_dataset = X_test_scaled, Y[test_idx]

            with open("processed_data/nyiso_raw.pkl", "wb") as f:
                pickle.dump((train_dataset, calibration_dataset, test_dataset), f, protocol=pickle.HIGHEST_PROTOCOL)

        with open("processed_data/nyiso_test_vis.pkl", "wb") as f:
            pickle.dump((X_test, Y[test_idx]), f, protocol=pickle.HIGHEST_PROTOCOL)

    return train_dataset, calibration_dataset, test_dataset


def _convert_daily_electricity_data() -> None:
    df = pd.read_csv('data/nyiso.csv')
    # Aggregating the dataset at daily level
    df['Timestamp'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')
    df.index = df['Timestamp']
    df = df.resample('D').mean()
    print(df)
    df.to_csv('data/nyiso_daily.csv')


def _convert_monthly_electricity_data() -> None:
    df = pd.read_csv('data/LoadData.csv')
    # Aggregating the dataset at monthly level
    df['Time'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')
    df.index = df['Time']
    df = df.resample('M').mean()
    # generate "year-month" format column
    df['Timestamp'] = df.index.strftime('%Y-%m')
    df = df.reset_index(drop=True)
    df.index = df['Timestamp']
    del df['Timestamp']
    print(df)
    df.to_csv('data/electricity_monthly.csv')

# if __name__ == '__main__':
    # convert_daily_electricity_data()
    # convert_monthly_electricity_data()
    # get_raw_electricity_data()
    # get_electricity_splits(length=100, horizon=50, conformal=True, n_train=6, n_calibration=3, n_test=2, cached=False, seed=None)
