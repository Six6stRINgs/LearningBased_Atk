from torch.utils.data import DataLoader
import torch.nn.functional as F
from blitz.modules import BayesianConv1d
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class CNN_LBA_Model(nn.Module):
    def __init__(self, features_cnt, windows_cnt, n):
        super(CNN_LBA_Model, self).__init__()
        self.conv1 = BayesianConv1d(windows_cnt, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = BayesianConv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * features_cnt, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2_classification = nn.Linear(64, (features_cnt * windows_cnt + 1))
        self.fc2_attack = nn.Linear(64, n)
        self.features_cnt = features_cnt
        self.windows_cnt = windows_cnt
        self.n = n

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * self.features_cnt)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)

        # classify the sensitive points(timestamp) where the attack occurs
        output_classification = self.fc2_classification(x)
        # predict the attack value of the corresponding sensitive points(timestamp)
        output_attack = self.fc2_attack(x)
        return output_classification, output_attack

    def __str__(self):
        return "CNN_LBA_Model"


def train_model(train_data, model, batch_size=25, learning_rate=0.005, epochs=50, device='cpu',
                print_info=False, plot_flag=False):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_atk = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_cls_list = []
    loss_atk_list = []

    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss_cls = 0.0
        total_loss_atk = 0.0

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        for batch in train_dataloader:
            inputs, label, value = batch
            inputs, label, value = inputs.to(device), label.to(device), value.to(device)

            outputs_cls, output_atk = model(inputs)

            optimizer.zero_grad()
            loss_cls = criterion_cls(outputs_cls, label.view(-1).long())
            loss_atk = criterion_atk(output_atk, value.float())
            loss_cls.backward(retain_graph=True)
            loss_atk.backward()
            optimizer.step()

            total_loss_cls += loss_cls.item()
            total_loss_atk += loss_atk.item()

        loss_cls_list.append(total_loss_cls / len(train_data))
        loss_atk_list.append(total_loss_atk / len(train_data))
        if print_info:
            print(f"Epoch {epoch + 1}/{epochs}, Loss_cls: {total_loss_cls / len(train_data)}"
                  f", Loss_atk: {total_loss_atk / len(train_data)}")

    if plot_flag:
        plt.plot(loss_cls_list)
        plt.title('Train Loss of Sensitive Point')
        plt.show()

        plt.plot(loss_atk_list)
        plt.title('Train Loss of Attack Perturbation')
        plt.show()
    return loss_cls_list, loss_atk_list


def test_model(test_data, model, print_info=True):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_atk = nn.MSELoss()

    model.eval()
    accuracy = 0
    total_loss_cls = 0.0
    total_loss_atk = 0.0
    with torch.no_grad():
        test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)
        for batch in test_dataloader:
            inputs, label, value = batch
            output_cls, output_atk = model(inputs)
            loss_cls = criterion_cls(output_cls, label.view(-1).long())
            loss_atk = criterion_atk(output_atk, value.float())
            total_loss_cls += loss_cls.item()
            total_loss_atk += loss_atk.item()
            for i in range(len(label)):
                point = torch.argmax(output_cls[i], dim=0).item()
                if point is label[i].item():
                    accuracy += 1

    if print_info:
        print(f"Test Loss Classification: {total_loss_cls / len(test_data)} accuracy: {accuracy / len(test_data)}"
              f", Loss Attack: {total_loss_atk / len(test_data)}")
