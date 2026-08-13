#Libraries
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import Animation
from matplotlib.colors import LogNorm
import numpy as np
import torch
#Types
from .types import TestingDomain, Solutions
from matplotlib.figure import Figure

FIG_SIZE_LONG = (15,4)
FIG_SIZE_SHORT = (8, 4)

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
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_LONG)

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

def animate_profiles(domain: TestingDomain,
                     solutions: Solutions,
                     which: tuple[str,...]=('predicted', 'exact'),
                     animation_len: int = 5,
                     save_path: str | None = None
                     ) -> Animation:

    x = domain.x_test.cpu().numpy()
    x_axis = x[:, 0]
    t = domain.t_test.cpu().numpy()
    t_axis = t[0,:]

    fig = plt.figure(figsize=FIG_SIZE_LONG)
    ax = fig.add_subplot(autoscale_on=False, xlim=(x_axis[0], x_axis[-1]),
                         ylim=(0, torch.max(solutions.predicted).item()+0.1))

    sol_field = []
    artists = []
    for sol_key in which:
        match sol_key:
            case 'exact': 
                sol_field.append(solutions.exact.cpu().numpy())
                temp, = ax.plot(x_axis, np.full_like(x_axis, np.nan), label=f'{sol_key}', linestyle='--', color='black')
                artists.append(temp)
            case 'linear': 
                sol_field.append(solutions.linear.cpu().numpy())
                temp, = ax.plot(x_axis, np.full_like(x_axis, np.nan), label=f'{sol_key}', linestyle=':', color='grey')
                artists.append(temp)
            case 'predicted': 
                sol_field.append(solutions.predicted.cpu().numpy())
                temp, = ax.plot(x_axis, np.full_like(x_axis, np.nan), label=f'{sol_key}', color='steelblue')
                artists.append(temp)
            case _: raise ValueError(f'Each key in which must be predicted, exact or linear.')

    
    ax.grid(True, alpha=0.4)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.legend()
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    total_frames = t_axis.size
    skip_step = max(1, total_frames // (animation_len * 30))
    frames = total_frames // skip_step
    fps = frames // animation_len
    print(f'Creating a GIF with an FPS of: {fps}')
    interval = (animation_len / frames) * 1000

    def animate(i):
        ret = []
        data_idx = i * skip_step
        which_idx = 0
        for profile in sol_field:
            artists[which_idx].set_ydata(profile[:,data_idx])
            ret.append(artists[which_idx])
            which_idx += 1 
        time_text.set_text(time_template % (t_axis[data_idx]))
        ret.append(time_text)

        return ret


    ani = animation.FuncAnimation(fig, animate, frames, interval=interval, blit=True)
    if save_path is not None: ani.save(save_path, writer=animation.PillowWriter(fps=fps))
    return ani
    

def plot_losses(components: list[str], 
                losses: dict[str, list[float]], 
                adam_epochs: int
                ) -> Figure:
    """
    Plots the losses of a given training run of a model
    """

    fig, ax = plt.subplots(figsize=FIG_SIZE_LONG)

    for comp in components:
        if comp in losses:
            ax.plot(losses[comp], label=f'{comp} loss')
        else:
            raise ValueError(f'Unknown loss component \'{comp}\'')
    
    if adam_epochs > 0: #does not include the optimizer switch if adam_epochs =< 0
        ax.axvline(x=adam_epochs, color='r', linestyle='--', alpha=0.7)
        ax.text(adam_epochs + 5, 0.2, 'Adam → L-BFGS', 
                rotation=90, verticalalignment='center', transform=plt.gca().get_xaxis_transform())

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
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_SHORT)


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

    fig, ax = plt.subplots(figsize=FIG_SIZE_SHORT)

    contour = ax.pcolormesh(t[0,:], x[:,0], error_plot, cmap='hot', norm=LogNorm())
    fig.colorbar(contour, ax=ax, label='Error')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Position (x)')
    fig.tight_layout()

    return fig
    
