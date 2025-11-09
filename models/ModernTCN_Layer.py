import torch
from torch import nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class FlattenHead(nn.Module):
    def __init__(self, individual: bool, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if individual:
            self.linears = nn.ModuleList([nn.Linear(nf, target_window) for _ in range(n_vars)])
            self.dropouts = nn.ModuleList([nn.Dropout(head_dropout) for _ in range(n_vars)])
        else:
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            outputs = []
            for i in range(self.n_vars):
                z = x[:, i, :, :].flatten(start_dim=-2)
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                outputs.append(z)
            return torch.stack(outputs, dim=1)
        x = x.flatten(start_dim=-2)
        x = self.linear(x)
        return self.dropout(x)
