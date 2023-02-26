from sklearn.preprocessing import StandardScaler
import numpy as np

import math

def standardize(x):
    mean = None
    var = None
    std = None
    z = None

    # 平均を算出、集合の総和 / 特徴数
    mean = sum(x) / len(x)

    # 分散を算出、集合の平均 - 特徴の差を2乗和したものを個数で割る
    var = sum([(mean - x_i) ** 2 for x_i in x]) / len(x)

    # 分散を平方根で割る
    std = math.sqrt(var)

    # 各特徴から平均を引き、それを標準偏差で割る
    z = [x_i / std for x_i in [x_i - mean for x_i in x]]

    return [mean, var, std, z]


def main():

    sample_data = np.array([[5.1], [4.9], [4.7], [4.6], [5.0], [5.4], [4.6]])
    mean, var, std, z = standardize(sample_data)
    print(f'mean = {mean}')
    print(f'var = {var}')
    print(f'std = {std}')
    print('z = ', end="")
    print(*z)

    print("===================================")

    scaler = StandardScaler()
    scaler.fit(sample_data)

    print(f'scaler.mean_ = {scaler.mean_}')
    print(f'scaler.var_ = {scaler.var_}')
    print(f'std = {math.sqrt(scaler.var_)}')
    print('z = ', end="")
    print(*scaler.transform(sample_data))

if __name__ == '__main__':
    main()