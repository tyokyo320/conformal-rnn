# Copyright (c) 2022, tyokyo320
# Licensed under the BSD 3-clause license

import pickle

import numpy as np
import pandas as pd
import torch
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
    if cached:
        with open("data/electricity.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        df = pd.read_csv("data/electricity.csv")
        print(df)

    return dataset


def get_electricity_splits(length=100, horizon=50, conformal=True, n_train=200, n_calibration=100, n_test=80, cached=True, seed=None) -> ElectricityDataset:
    pass


def convert_daily_electricity_data() -> None:
    df = pd.read_csv('data/LoadData.csv')
    # Aggregating the dataset at daily level
    df['Timestamp'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')
    df.index = df['Timestamp']
    df = df.resample('D').mean()
    print(df)
    df.to_csv('data/electricity_daily.csv')


def convert_monthly_electricity_data() -> None:
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

if __name__ == '__main__':
    # convert_daily_electricity_data()
    convert_monthly_electricity_data()





