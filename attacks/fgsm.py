import torch


class FGSM:
    def __init__(self, eps, model, loss_type="MSE") -> None:
        self.eps = eps
        self.model = model

        if loss_type == "MSE":
            self.criterion = torch.nn.MSELoss(reduction="mean")
        elif loss_type == "L1":
            self.criterion = torch.nn.L1Loss(reduction="mean")
        else:
            raise Exception("Loss function" + str(loss_type) + "is not \"MSE\" or \"L1\"")

    def attack(self, X, target, window_range):
        X_adv = torch.clone(X).detach()
        X_adv.requires_grad = True
        y_pred = self.model(X_adv)

        loss = self.criterion(y_pred, target)

        self.model.zero_grad()
        loss.backward()
        data_grad = X_adv.grad.data
        sign_data_grad = data_grad.sign()
        X_adv = X_adv + self.eps * sign_data_grad * window_range

        return X_adv

    def __str__(self) -> str:
        return "FGSM"
