import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from cnn import CNN
from lstm import LSTM
from gru import GRU

from data_provider import get_ori_data
from utils import check_file_existence


def train_model(model, lr, num_epochs, x_train, y_train, device, batch_size, print_time=False):
    model.to(device)
    model.train()
    criterion = torch.nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert to TensorDataset
    train_data = TensorDataset(x_train, y_train)
    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_train_pred = model(x_batch)
            loss = criterion(y_train_pred, y_batch)
            losses[epoch] = loss.item()

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if print_time:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses[epoch]}')
    cost_time = time.time() - start_time

    if print_time:
        print(f'Training time of {model} is: {cost_time} Sec.')

    return losses


def test_model(model, x_test, y_test, device):
    model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss(reduction="mean")
    test_data = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            loss = criterion(y_test_pred, y_batch)
            losses.append(loss.item())
    return losses


def train_models_based_on_paras(paras, data, path_root, plot_loss=True):
    model = None
    lr = None
    epoch = paras["epochs"]
    device = paras["device"]
    batch = paras["batch_size"]
    out = paras["output_dim"]
    for model_name in ["CNN", "LSTM", "GRU"]:
        if model_name == "CNN":
            lr, window_size, c, f0, f1, f2 = paras[model_name][data.df_name]
            model = CNN(window_size, c, f0, f1, f2, out)
        elif model_name == "LSTM":
            lr, input_dim, hidden_dim, num_layers = paras[model_name][data.df_name]
            model = LSTM(input_dim, hidden_dim, num_layers, out)
        elif model_name == "GRU":
            lr, input_dim, hidden_dim, num_layers = paras[model_name][data.df_name]
            model = GRU(input_dim, hidden_dim, num_layers, out)

        loss = train_model(model, lr, epoch, data.X_train, data.Y_train, device, batch, print_time=True)
        save_model(path_root, data.df_name, data.seed, model)
        print(f"Model {model_name} for {data.df_name} is trained and saved.")

        if plot_loss:
            plt.plot(loss)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_name} Loss for {data.df_name}')
            plt.show()


def save_model(path_root, df_name, seed, model):
    path = os.path.join(path_root, 'saved_models', f'df_{df_name}_model_{model}_seed_{seed}.pkl')
    if not check_file_existence(path):
        with open(path, "wb"):
            torch.save(model, path)


def load_model(path_root, df_name, seed, model):
    path = os.path.join(path_root, 'saved_models', f'df_{df_name}_model_{model}_seed_{seed}.pkl')
    with open(path, "rb"):
        result = torch.load(path)
        result.eval()
    return result


# train the dataset, run the following code
if __name__ == '__main__':
    paras_dict = {
        "CNN": {
            "Electricity": [0.005, 4, 256, 3, 256, 128],
            "NZTemp": [0.01, 3, 64, 9, 64, 64],
            "CNYExch": [0.005, 7, 256, 5, 256, 128],
            "Oil": [0.01, 7, 256, 6, 256, 32]
        },
        "LSTM": {
            "Electricity": [0.01, 3, 64, 1],
            "NZTemp": [0.01, 9, 256, 1],
            "CNYExch": [0.01, 5, 64, 2],
            "Oil": [0.01, 6, 128, 1]
        },
        "GRU": {
            "Electricity": [0.005, 3, 256, 1],
            "NZTemp": [0.005, 9, 256, 4],
            "CNYExch": [0.01, 5, 64, 1],
            "Oil": [0.01, 6, 128, 4]
        },
        "output_dim": 1,
        "batch_size": 64,
        "epochs": 100,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": 2210
    }
    PATH_ROOT = os.path.dirname(os.getcwd())

    for data in ["Electricity", "NZTemp", "CNYExch", "Oil"]:
        ori_data = get_ori_data(data, PATH_ROOT, seed=paras_dict['seed'])
        train_models_based_on_paras(paras_dict, ori_data, PATH_ROOT, plot_loss=True)
