#Libraries
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
#Types
from .types import TestingDomain, Solutions
from matplotlib.figure import Figure

FIG_SIZE = (15,4)

def plot_profiles(t_values: list[int], 
                  domain: TestingDomain,
                  solutions: Solutions, 
                  which: tuple[str, ...]=("predicted",)
                  ) -> Figure:
    """
    Plots the specified profiles of a given set of solutions
    """
    
    x = domain.x_test.cpu().numpy()
    t = domain.t_test.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for sol_key in which:
        match sol_key:
            case 'exact': sol_field = solutions.exact.cpu().numpy()
            case 'linear': sol_field = solutions.linear.cpu().numpy()
            case 'predicted': sol_field = solutions.predicted.cpu().numpy()
            case _: raise ValueError(f'Each key in which must be predicted, exact or linear.')
        
        t_axis = t[0, :]
        indices = [int(np.argmin(np.abs(t_axis - t_val))) for t_val in t_values]
        x_axis = x[:, 0]
        profiles = [sol_field[:, idx] for idx in indices]

        for t_val, profile in zip(t_values, profiles):
            ax.plot(x_axis, profile, label=f'{sol_key} t= {t_val}')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()        

    return fig


def plot_losses(components: list[str], 
                losses: dict[str, list[float]], 
                adam_epochs: int
                ) -> Figure:
    """
    Plots the losses of a given training run of a model
    """

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for comp in components:
        if comp in losses:
            ax.plot(losses[comp], label=f'{comp} loss')
        else:
            raise ValueError(f'Unknown loss component \'{comp}\'')
    
    if adam_epochs > 0: #does not include the optimizer switch if adam_epochs =< 0
        ax.axvline(x=adam_epochs, color='r', linestyle='--', alpha=0.7)
        ax.text(adam_epochs + 5, 0.2, 'Adam → L-BFGS', 
                rotation=90, verticalalignment='center', transform=ax.gca().get_xaxis_transform())

    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.tight_layout()
    
    return fig


def plot_spacetime(domain: TestingDomain,
                   u_pred: torch.Tensor,
                   scatter_coords: dict[str, torch.Tensor] | None = None
                   ) -> Figure:
    """
    Plots the specified spacetime mesh of a given predicted solution, a scatter function
    is avaible to show the different coordinates used in training
    """

    x = domain.x_test.cpu().numpy()
    t = domain.t_test.cpu().numpy()
    u = u_pred.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)


    contour = ax.pcolormesh(t[0,:], x[:,0], u, cmap='plasma', shading='auto')
    fig.colorbar(contour, ax=ax, label='u(x,t)')

    if scatter_coords is not None:

        settings = {
            'boundary': ('.', 'red', 1),
            'initial': ('x', 'white', 3),
            'pde': ('.', 'black', 0.3)
        }

        for key, coords in scatter_coords.items():
            scatter_x = coords[0].cpu().numpy()
            scatter_t = coords[1].cpu().numpy()
            ax.scatter(scatter_t, scatter_x, marker=settings[key][0], color=settings[key][1],
                        alpha=0.5, s=settings[key][2], label=key)
        
        ax.legend(loc='upper right', fontsize='small')

    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Position (x)')
    fig.tight_layout()

    return fig

def plot_heatmap(error: torch.Tensor,
                 domain: TestingDomain
                 ) -> Figure:
    """
    Plots the error of a model as a heatmap
    """
    
    x = domain.x_test.cpu().numpy()
    t = domain.t_test.cpu().numpy()
    error_plot = error.cpu().numpy()

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    contour = ax.pcolormesh(t[0,:], x[:,0], error_plot, cmap='hot', norm=LogNorm())
    fig.colorbar(contour, ax=ax, label='Error')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Position (x)')
    fig.tight_layout()

    return fig
    
