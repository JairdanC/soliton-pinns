import torch
from kdv_types import ErrorStats
from kdv_types import TestingDomain

def setup_testing_domain(x_lims: torch.Tensor, 
                         t_lims: torch.Tensor,
                         nx: int = 1000, nt: int = 1000
                         ) -> TestingDomain:
    """
    Setup the testing domain on a regular grid from on the x and t limits
    with nx and nt collocation points respectively, return a TestingDomain
    dataclass
    """


    x0 = x_lims[0]
    x1 = x_lims[1]
    t0 = t_lims[0]
    t1 = t_lims[1]

    x_axis = torch.linspace(x0, x1, nx)
    t_axis = torch.linspace(t0, t1, nt)

    X, T = torch.meshgrid(x_axis, t_axis, indexing='ij')

    domain = TestingDomain(X, T)
    return domain

def test(u_pred: torch.Tensor, 
         u_exact: torch.Tensor,
         error_type: str = 'absolute-normalized', 
         verbose: bool = True
         ) -> ErrorStats:
    """
    Find the absolute-normalized (or absolute) difference between the exact and
    predicted solution over the regular testing grid, return as a ErrorStats dataclass
    """

    if error_type == 'absolute':
        error = torch.abs(u_pred - u_exact)
    elif error_type == 'absolute-normalized':
        max_exact = torch.max(torch.abs(u_exact))
        error = torch.abs(u_pred - u_exact) / max_exact
    else:
        raise ValueError(f'invalid error type: {error_type}')
    
    mae = torch.mean(error).item()
    max_error = torch.max(error).item()

    error_stats = ErrorStats(mae, max_error, error)

    if verbose:
        print(f"{error_type} error metrics:")
        print(f"Mean: {mae:.6e}")
        print(f"Maximum: {max_error:.6e}")

    return error_stats