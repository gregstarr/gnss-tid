import logging

import numpy as np
import torch
from torch import nn


class CorrelationModel(nn.Module):
    def __init__(self, c0, w) -> None:
        super().__init__()
        self.history = {
            "center": [],
            "wavelength": [],
            "metric": [],
            "phase": [],
        }
        self.center = nn.Parameter(torch.tensor(c0))
        self.wavelength = nn.Parameter(torch.tensor(w))
    
    def get_result(self, epoch):
        wavelength = self.history["wavelength"][epoch]
        offset = self.history["phase"][epoch] * wavelength / (2 * np.pi)
        return {
            "center": self.history["center"][epoch],
            "wavelength": wavelength,
            "metric": self.history["metric"][epoch],
            "offset": offset,
            "history": self.history,
        }

    def forward(self, xy, tid):
        dist = torch.linalg.vector_norm(xy - self.center, dim=1, keepdim=True)
        phase = 2.0 * torch.pi * dist / self.wavelength
        phase_offset_vals = torch.linspace(0.0, 2.0 * torch.pi, 25)
        model_out = torch.cos(phase + phase_offset_vals[None, :])
        
        vals = torch.mean(model_out * tid[:, None], dim=0)
        metric = torch.max(vals)
        phase = phase_offset_vals[torch.argmax(vals)]

        self.history["center"].append(self.center.clone().detach().numpy())
        self.history["wavelength"].append(self.wavelength.item())
        self.history["metric"].append(metric.item())
        self.history["phase"].append(phase.item())

        return metric


class EarlyStopping:
    def __init__(self, improvement, period) -> None:
        self.best = None
        self.best_epoch = 0
        self.improvement = improvement
        self.period = period
    
    def should_stop(self, val, epoch):
        if self.best is None:
            self.best = val
            self.best_epoch = epoch
            return False
        
        check_val = 2 * (val - self.best) / abs(val + self.best)
        if check_val > self.improvement:
            self.best_epoch = epoch
            self.best = val
        
        if epoch - self.best_epoch > self.period:
            return True
        return False


class CenterFinder:
    def __init__(self, max_iter, learning_rate, improvement, steps):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.stopper = EarlyStopping(improvement, steps)

    def __call__(self, c0, w, x, y, tec):
        model = CorrelationModel(c0, w)
        r = torch.tensor(np.column_stack([x, y]))
        tid = torch.tensor(tec)

        optimizer = torch.optim.Adam(
            model.parameters(), self.learning_rate, maximize=True
        )
        for epoch in range(self.max_iter):
            model.zero_grad()
            metric = model(r, tid)
            metric.backward()
            optimizer.step()
            if self.stopper.should_stop(metric.item(), epoch):
                break
        
        return model.get_result(self.stopper.best_epoch)
