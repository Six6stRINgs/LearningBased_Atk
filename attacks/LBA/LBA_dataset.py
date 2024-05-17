from torch.utils.data import Dataset
import torch


# Dataset of LBA: the adversarial examples
class LBA_Dataset(Dataset):
    def __init__(self, device='cpu', data=None, labels=None, value=None):
        self.data = torch.empty(0).to(device) if data is None else data
        self.labels = torch.empty(0).to(device) if labels is None else labels
        self.value = torch.empty(0).to(device) if value is None else value

    def add_X(self, x):
        self.data = torch.cat((self.data, x), dim=0)

    def add_Y(self, y):
        self.labels = torch.cat((self.labels, y), dim=0)

    def add_value(self, value):
        self.value = torch.cat((self.value, value), dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        value = self.value[idx]
        return x, y, value

    def __str__(self):
        return "LBA_Dataset"


def get_sensitive_point_and_value(eta, n):
    # eta -> tensor(1,windows_cnt,class_cnt)
    windows_cnt = eta.shape[1]
    class_cnt = eta.shape[2]
    tmp = eta.clone().transpose(1, 2).reshape(1, -1)
    points = torch.empty(0)
    values = torch.empty(0)
    for ind, sub in enumerate(tmp[0]):
        if sub != 0:
            points = torch.cat((points, torch.tensor([[ind]])), dim=1)
            values = torch.cat((values, torch.tensor([[sub]])), dim=1)

    if points is torch.empty(0):
        points = torch.tensor([[class_cnt * windows_cnt]])
        values = torch.tensor([[0]])
    while points.shape[1] < n:
        points = torch.cat((points, torch.tensor([[0]])), dim=1)
        values = torch.cat((values, torch.tensor([[windows_cnt * class_cnt]])), dim=1)
    return points, values


def get_total_sensitive_point_from_adv(features_cnt, windows_cnt, ori_data, X_adv, n):
    sp_statistics = [0] * (features_cnt * windows_cnt + 1)
    sp_location = []
    data_cnt = ori_data.X_test.shape[0]
    for i in range(data_cnt):
        X_current = ori_data.X_test[i].unsqueeze(0)
        X_current_adv = X_adv[i].unsqueeze(0)
        eta = X_current_adv - X_current
        points, _ = get_sensitive_point_and_value(eta, n)
        for po in points[0]:
            sp_statistics[po.int().item()] += 1
        sp_location.append(points[0].numpy().tolist())
    return sp_statistics, sp_location


def get_LBA_Dataset(ori_data, X_adv, adv_cnt, n, device):
    LBA_data = LBA_Dataset(device=device)
    for test_ind in range(adv_cnt):
        X_current = ori_data.X_test[test_ind].unsqueeze(0)
        X_current_adv = X_adv[test_ind].unsqueeze(0)

        point, values = get_sensitive_point_and_value(X_current_adv - X_current, n)
        LBA_data.add_X(X_current.to(device))
        LBA_data.add_Y(point.to(device))
        LBA_data.add_value(values.to(device))

    return LBA_data
