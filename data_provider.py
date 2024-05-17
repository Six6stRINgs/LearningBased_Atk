import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TSData:
    def __init__(self, df_path=None, df_name=None, y_col_name=None, window_size=None, seed=None):

        self.X_train = None
        self.Y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.Y_test = None
        self.seed = None
        if seed is not None:
            self.seed = int(seed)
        self.df_name = str(df_name)
        self.y_col_name = str(y_col_name)
        self.y_ind = None
        self.window_size = window_size
        self.window_ranges = []
        self.df_path = df_path
        self.single_X_shape = None
        self.single_y_shape = None

        self.train_valid_test_split(250, 0.2)

    def normalize_data(self, feature_range=(0, 1), standard=True):
        """
        Normalize data based on train set
        """
        if standard:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler(feature_range=feature_range)

        for col_ind in range(self.X_train.shape[2]):
            self.X_train[:, :, col_ind] = scaler.fit_transform(self.X_train[:, :, col_ind].reshape(-1, 1)).reshape(
                self.X_train[:, :, col_ind].shape)
            self.X_valid[:, :, col_ind] = scaler.transform(self.X_valid[:, :, col_ind].reshape(-1, 1)).reshape(
                self.X_valid[:, :, col_ind].shape)
            self.X_test[:, :, col_ind] = scaler.transform(self.X_test[:, :, col_ind].reshape(-1, 1)).reshape(
                self.X_test[:, :, col_ind].shape)
            if col_ind == self.y_ind:
                self.Y_train = scaler.transform(self.Y_train)
                self.y_valid = scaler.transform(self.y_valid)
                self.Y_test = scaler.transform(self.Y_test)

    def calculate_test_window_ranges(self):
        for i in range(self.X_test.shape[0]):
            single_test_window_range = []
            for f in range(self.X_test.shape[2]):
                single_test_window_range.append(np.ptp(self.X_test[i: i + 1, :, f: f + 1]))
            self.window_ranges.append(torch.FloatTensor(single_test_window_range))

    def get_single_shape(self, is_X):
        """
        return single X shape if is_X is true
        return single Y shape otherwise
        """
        if is_X:
            single_shape = list(self.X_test.shape)
            single_shape[0] = 1
            self.single_X_shape = single_shape
        else:
            single_shape = list(self.Y_test.shape)
            single_shape[0] = 1
            self.single_y_shape = single_shape

        return single_shape

    def train_valid_test_split(self, test_size, valid_per):
        """
        Split the data into three sets
        Obtain the self.y_ind (the index of y)
        """
        df = pd.read_csv(self.df_path, sep=",")

        self.y_ind = df.columns.get_loc(self.y_col_name)

        raw_X_data = df.to_numpy()
        raw_y_data = df[self.y_col_name].to_numpy().reshape(df[self.y_col_name].shape[0], 1)

        X_data, y_data = [], []
        for index in range(len(raw_X_data) - self.window_size - 1):
            X_data.append(raw_X_data[index: index + self.window_size])
            y_data.append(raw_y_data[index + self.window_size])

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=test_size,
                                                                    random_state=self.seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=valid_per,
                                                              random_state=self.seed)

        self.X_train = X_train
        self.Y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.Y_test = y_test

        self.normalize_data((0, 1), False)
        self.calculate_test_window_ranges()
        self.get_single_shape(True)
        self.get_single_shape(False)

        self.X_train = torch.from_numpy(self.X_train).type(torch.Tensor)
        self.Y_train = torch.from_numpy(self.Y_train).type(torch.Tensor)
        self.X_valid = torch.from_numpy(self.X_valid).type(torch.Tensor)
        self.y_valid = torch.from_numpy(self.y_valid).type(torch.Tensor)
        self.X_test = torch.from_numpy(self.X_test).type(torch.Tensor)
        self.Y_test = torch.from_numpy(self.Y_test).type(torch.Tensor)

    def __str__(self) -> str:
        return self.df_name


def get_ori_data(dataset: str, path_root: str, seed: int) -> TSData:
    df_path = os.path.join(path_root, "datasets", f'{dataset}.csv')
    if dataset == 'Electricity':
        data = TSData(df_path, dataset, 'Consumption', 4, seed)
    elif dataset == 'NZTemp':
        data = TSData(df_path, dataset, 'Auckland', 3, seed)
    elif dataset == 'CNYExch':
        data = TSData(df_path, dataset, 'Close', 7, seed)
    elif dataset == 'Oil':
        data = TSData(df_path, dataset, 'Close', 7, seed)
    else:
        raise ValueError("Invalid dataset name")
    return data
