import ast
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_csv_line_from_list(line_list):
    """
    Create a comma seperated line without new line character with given list
    """
    line_str = ""
    for value in line_list:
        line_str += str(value)
        line_str += ","
    return line_str


def create_empty_result_csv_file(path_save, first_line_list):
    if not check_file_existence(path_save):
        line = get_csv_line_from_list(first_line_list)
        with open(path_save, "a") as f:
            f.write(line)
            f.write("\n")


def append_result_to_csv_file(path_save, line_list):
    line = get_csv_line_from_list(line_list)
    with open(path_save, "a") as f:
        f.write(line)
        f.write("\n")


def check_result_file_path(path_file):
    """
    while the path_file exists, add 1 before .csv in the path_file
    """
    while os.path.exists(path_file):
        path_file = Path(str(path_file)[:-4] + "(1).csv")
    return path_file


def set_seed(random_state=None):
    """Reset RNG seed."""
    if random_state is None:
        random_state = random.randint(1, 999999)
    random_state = int(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    return random_state


def check_file_existence(path, raise_error=False):
    """
    Check file existence with given path
    """
    bool_result = os.path.exists(path)
    if bool_result:
        if raise_error:
            raise Exception(f'File {path} already exists!')
        else:
            print(f'File {path} already exists!')
    return bool_result


def get_tensor(df_data) -> torch.tensor:
    tensor = df_data.replace(';', ',')
    tensor = ast.literal_eval(tensor)
    tensor = torch.tensor(tensor)
    return tensor


def get_Adv_from_csv(csv_path):
    """
    As we get data by the name of the column name, it is necessary to keep the column name consistent.
    """
    df = pd.read_csv(csv_path)
    x_adv = torch.empty(0)
    y_adv = torch.empty(0)
    y_pred = torch.empty(0)

    for i in range(df['Adv Example'].size):
        X_adv_temp = get_tensor(df['Adv Example'][i])
        Y_adv_temp = torch.tensor(df['Attacked Y Pred'][i]).unsqueeze(0)
        Y_pred_temp = torch.tensor(df['Original Y Pred'][i]).unsqueeze(0)

        x_adv = torch.cat((x_adv, X_adv_temp), 0)
        y_adv = torch.cat((y_adv, Y_adv_temp), 0)
        y_pred = torch.cat((y_pred, Y_pred_temp), 0)
    return x_adv, y_adv, y_pred


def check_incorrect_cnt_of_NVITA(eta, n):
    """
        eta: tensor(x, windows_cnt, features_cnt)
        During experiments, we found that the NVITA does not always generate the n perturbations.
    """
    res = 0
    for i in range(eta.shape[0]):
        cnt = 0
        tmp = eta[i].clone().view(-1)
        for j in range(tmp.shape[0]):
            if tmp[j] != 0:
                cnt = cnt + 1
        if cnt < n:
            res = res + 1
    return res


def tensor_to_csv(data: torch.tensor, path: str, columns_name: list[str]) -> None:
    # data: a 3-D tensor
    cnt = data.shape[0]
    window_cnt = data.shape[1]
    class_cnt = data.shape[2]
    df = pd.DataFrame(data.view(cnt * window_cnt, class_cnt).numpy(), columns=columns_name)
    df.to_csv(path, index=False)
    print(f"Save the tensor data to {path}")


def df_normalize(df: pd.DataFrame, norm_range: tuple = (0, 1)) -> pd.DataFrame:
    columns = df.columns
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * \
                  (norm_range[1] - norm_range[0]) + norm_range[0]
    return df
