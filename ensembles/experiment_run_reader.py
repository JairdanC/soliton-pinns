from pathlib import Path
import sys
import numpy as np
import gc

root = Path(__file__).parent

def model_error_calculation(model: str = 'kdv-1-soliton', error_type: str = 'absolute-normalized', verbose: bool = True):
    """
    Recovers the results from the save_experiment_run call in driver-* for 
    various configurations of the PINN, specifically the mean and max errors
    for the type of error used in the train() call of the driver

    model: str | Path
    the ensemble/replication folder with the driver which has been run, 
    expecting errors folder containing the grid wise error for each point 
    on the test grid

    error_type: str
    this is for continuity purposes, not required for the function to operate
    only so that the user is sure of which error they are calculating from
    the .npz files, should match the parameter called in test() for the driver
    [absolute-normalized is the default]
    """

    #initialize the file 
    path = Path(root / 'ensembles' / 'replications' / model / 'errors')
    model_mean = []
    model_max = []

    for file in path.iterdir():
        if file.is_file(): #Filter out any subdirectories
            data = np.load(file)
            model_mean.append(np.mean(data))
            model_max.append(np.max(data))
    
    if (verbose):
        print(f'{error_type} error mean over all seeds: ' + str(np.mean(model_mean)))
        print(f'{error_type} error maximum mean over all seeds: ' + str(np.max(model_mean)))
        print(f'{error_type} mean max error over all seeds: ' + str(np.mean(model_max)))
        print(f'{error_type} max error over all seeds: ' + str(np.max(model_max)))

    return (model_mean, model_max)

def training_time(model: str = 'kdv-1-soliton', verbose: bool = True): 
    return False #not implemented yet






