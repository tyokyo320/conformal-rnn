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
    

def get_raw_electricity_data(cached=True):
    if cached:
        with open("data/electricity.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = []
        df = pd.read_csv("data/electricity.csv")
        print(df)


def get_electricity_splits():
    pass


def convert_daliy_electricity_data() -> None:
    df = pd.read_csv('data/LoadData.csv')
    # Aggregating the dataset at daily level
    df['Timestamp'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')
    df.index = df['Timestamp']
    df = df.resample('D').mean()
    print(df)
    df.to_csv('data/electricity.csv')


if __name__ == '__main__':
    convert_daliy_electricity_data()





