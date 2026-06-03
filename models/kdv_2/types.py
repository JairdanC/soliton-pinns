import torch
from dataclasses import dataclass

@dataclass(slots=True)
class TrainingDomain():
    x_coll: torch.Tensor
    t_coll: torch.Tensor
    x_ic: torch.Tensor
    t_ic: torch.Tensor
    u_ic: torch.Tensor
    x_bc: torch.Tensor
    t_bc: torch.Tensor
    u_bc: torch.Tensor

@dataclass(frozen=True, slots=True)
class TestingDomain():
    x_test: torch.Tensor
    t_test: torch.Tensor


@dataclass(frozen=True, slots=True)
class Solutions():
    exact: torch.Tensor
    linear: torch.Tensor
    predicted: torch.Tensor

@dataclass(frozen=True, slots=True)
class ErrorStats():
    mae: float
    max_error: float
    error: torch.Tensor

@dataclass(frozen=True, slots=True)
class TrainingStats():
    losses: dict[str, list[float]]
    time: float