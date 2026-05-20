import matplotlib.pyplot as plt
import numpy as np
import torch

FIG_SIZE = (15,4)

def plot_profiles(t_values: list[int], x_test: torch.Tensor, t_test: torch.Tensor, solutions: dict[str, torch.Tensor], which: tuple[str, ...]=("predicted",)) -> None:
    
    x = x_test.cpu().numpy()
    t = t_test.cpu().numpy()
    
    plt.figure(figsize=FIG_SIZE)

    for sol_key in which:
        if sol_key not in ('predicted', 'exact', 'linear'):
            raise ValueError(f'Each key in which must be predicted, exact or linear.')
        t_axis = t[0, :]
        indices = [int(np.argmin(np.abs(t_axis - t_val))) for t_val in t_values]
        sol_field = solutions[sol_key].cpu().numpy()
        profiles = [sol_field[:, idx] for idx in indices]
        x_axis = x[:, 0]

        for t_val, profile in zip(t_values, profiles):
            plt.plot(x_axis, profile, label=f'{sol_key} t= {t_val}')

    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()        

    return 


def plot_losses(components: list[str], losses: dict[str, list[torch.Tensor]], adam_epochs: int) -> None:

    plt.figure(figsize=FIG_SIZE)

    for comp in components:
        if comp in losses:
            plt.plot(losses[comp], label=f'{comp} loss')
        else:
            raise ValueError(f'Unknown loss component \'{comp}\'')
    
    if adam_epochs > 0: #does not include the optimizer switch if adam_epochs =< 0
        plt.axvline(x=adam_epochs, color='r', linestyle='--', alpha=0.7)
        plt.text(adam_epochs + 5, 0.2, 'Adam → L-BFGS', 
                rotation=90, verticalalignment='center', transform=plt.gca().get_xaxis_transform())

    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    
    return

# spacetime
def plot_spacetime(x_test: torch.Tensor, t_test: torch.Tensor, u_pred: torch.Tensor) -> None:
    x = x_test.cpu().numpy()
    t = t_test.cpu().numpy()
    u = u_pred.cpu().numpy()
    
    plt.figure(figsize=FIG_SIZE)


    contour = plt.pcolormesh(t[0,:], x[:,0], u, cmap='plasma', shading='auto')
    plt.colorbar(contour, label='u(x,t)')

    #missing scatter function, will restore later
    
    plt.xlabel('Time (t)')
    plt.ylabel('Position (x)')
    plt.tight_layout()

    return