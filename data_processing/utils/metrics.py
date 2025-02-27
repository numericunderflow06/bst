import torch


def wmape_metric(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(pred - true), dim=0) / torch.sum(true, dim=0)
