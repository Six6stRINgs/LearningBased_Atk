import torch


class BIM:
    def __init__(self, eps, alpha, steps, model, loss_type="MSE") -> None:
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
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
        min_X_adv = X - self.eps * window_range
        max_X_adv = X + self.eps * window_range

        for _ in range(self.steps):
            y_pred = self.model(X_adv)
            self.model.zero_grad()
            loss = self.criterion(y_pred, target)
            loss.backward()
            data_grad = X_adv.grad.data
            sign_data_grad = data_grad.sign()
            X_adv = torch.clamp(X_adv + self.alpha * sign_data_grad * window_range, min_X_adv, max_X_adv).detach()
            X_adv.requires_grad = True
        return X_adv

    def __str__(self) -> str:
        return "BIM"
