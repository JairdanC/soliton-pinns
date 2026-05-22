import torch
from kdv import ErrorStats

def setup_testing_domain(nx=1000, nt=1000):

    return

def test(u_pred: torch.Tensor, u_exact: torch.Tensor,
         error_type: str = 'absolute-normalized', verbose: bool = True
         ) -> ErrorStats:

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