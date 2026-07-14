#Libraries
import torch
import numpy as np
from scipy.signal import find_peaks
#Types
from .types import ErrorStats, TestingDomain

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

def compute_x_diff(u_pred: np.ndarray,
                   u_exact: np.ndarray,
                   n_solitons: int,
                   x: np.ndarray
                   ) -> np.ndarray:
    """
    Computes the x difference between soliton peaks, scaled to the domain
    ----
    Picks out the n peaks of the solitons of both the exact and predicted solutions. 
    Takes the corresponding values in the x domain, computes the x difference
    """

    try:  
        exact_peak_idx, _ = find_n_peaks(u_exact, num=n_solitons, distance=1)
        pred_peak_idx, _ = find_n_peaks(u_pred, num=n_solitons, distance=1)

        if exact_peak_idx.size != pred_peak_idx.size:
            raise ValueError("Different numbers of peaks found")
        
    except ValueError:
        error =  np.ones(n_solitons) * np.nan
    else:
        scale = x[-1] - x[0] 
        diff = np.abs(x[exact_peak_idx] - x[pred_peak_idx]) / scale
        error = np.ones(n_solitons) * np.nan
        error[:diff.size] = diff
    finally:
        return error

def compute_amp_diff(u_pred: np.ndarray,
                     u_exact: np.ndarray,
                     n_solitons: int
                     ) -> np.ndarray:
    """
    Computes the amplitude difference between soliton peaks, scaled to the domain
    ----
    Picks out the n peaks of the solitons of both the exact and predicted solutions. 
    Takes the corresponding values in the u domain, computes the amplitude difference
    """

    try:  
        exact_peak_idx, _ = find_n_peaks(u_exact, num=n_solitons, distance=1)
        pred_peak_idx, _ = find_n_peaks(u_pred, num=n_solitons, distance=1)

        if exact_peak_idx.size != pred_peak_idx.size:
            raise ValueError("Different numbers of peaks found")
        
    except ValueError:
        error =  np.ones(n_solitons) * np.nan
    else:
        scale = np.max(np.abs(u_exact)) 
        diff = np.abs(u_exact[exact_peak_idx] - u_pred[pred_peak_idx]) / scale
        error = np.ones(n_solitons) * np.nan
        error[:diff.size] = diff
    finally:
        return error

def find_n_peaks(signal: np.ndarray, num: int, distance: int) -> np.ndarray:
        "Recursive helper function to find the correct number of soliton peaks"
        if distance > signal.size:
            raise ValueError("Distance > signal")
        
        peak_idx, _ = find_peaks(signal, distance=distance)
        if peak_idx.size > num:
            peak_idx, _ = find_n_peaks(signal=signal, num=num, distance=distance+1)
            return peak_idx, _
        else:
            return peak_idx, _