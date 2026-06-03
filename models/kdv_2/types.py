import torch
from dataclasses import dataclass

@dataclass
class TrainingDomain(slots=True):
    x_coll: torch.Tensor
    t_coll: torch.Tensor
    x_ic: torch.Tensor
    t_ic: torch.Tensor
    u_ic: torch.Tensor
    x_bc: torch.Tensor
    t_bc: torch.Tensor
    u_bc: torch.Tensor

@dataclass
class TestingDomain(frozen=True, slots=True):
    x_test: torch.Tensor
    t_test: torch.Tensor


@dataclass
class Solutions(frozen=True, slots=True):
    exact: torch.Tensor
    linear: torch.Tensor
    predicted: torch.Tensor

@dataclass
class ErrorStats(frozen=True, slots=True):
    mae: float
    max_error: float
    error: torch.Tensor

@dataclass
class TrainingStats(frozen=True, slots=True):
    losses: dict[str, list[float]]
    time: float