import math
import pandas as pd


def get_MAE_ori(df: pd.DataFrame) -> tuple[float, float]:
    mae_atk = 0.0
    mae_ori = 0.0
    cnt = df['Attacked AE'].size
    for i in range(cnt):
        mae_atk += df['Attacked AE'][i]
        mae_ori += df['Original AE'][i]
    return mae_atk / cnt, mae_ori / cnt


def get_RMSE_ori(df: pd.DataFrame) -> tuple[float, float]:
    mse_atk = 0.0
    mse_ori = 0.0
    cnt = df['Attacked AE'].size
    for i in range(cnt):
        mse_atk += df['Attacked AE'][i] ** 2
        mse_ori += df['Original AE'][i] ** 2
    return math.sqrt(mse_atk / cnt), math.sqrt(mse_ori / cnt)


def get_SD_ori(df: pd.DataFrame) -> float:
    sd = 0.0
    cnt = df['True y'].size
    mean = df['True y'].mean()
    for i in range(cnt):
        sd += (df['True y'][i] - mean) ** 2
    return math.sqrt(sd / cnt)


def get_RSE_ori(df: pd.DataFrame) -> tuple[float, float]:
    rmse_atk, rmse_ori = get_RMSE_ori(df)
    sd = get_SD_ori(df)
    return rmse_atk / sd, rmse_ori / sd


def get_AR_ori(df: pd.DataFrame) -> float:
    ar = 0.0
    cnt = df['Cost Time'].size
    for i in range(cnt):
        ar += df['Cost Time'][i]
    return ar / cnt
