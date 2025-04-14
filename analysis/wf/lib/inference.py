"""
All functions related to actually calculating concentrations.
"""
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from .model import y_model
from .optimizer_lib import OptimizerWithEarlyStopping, run_minimize_with_stopping
from .fitting_lib import multi_sample_constraints, multi_sample_error, single_sample_constraints, single_sample_error
from .constants import NUM_SHARED_PARAMS

def fit_shared_params(
    data: list,
    num_iter: int = 3,
    tol: float = 0.5,
    threshold: float = 5,
) -> list:
    """
    Runs the initial shared parameter fitting process with the provided data.
    """
    assert len(data) > 0

    # Infer from data
    num_samples = len(data)
    num_genes = len(data[0])
    
    # Collect parameters and errors
    param_res = []
    
    for iter_ind in range(num_iter):
        print(f"Beginning shared fitting iteration {iter_ind + 1} of {num_iter}...")
        # Shared params + DNA and DNA_excess for each gene in each sample
        initial_params = np.concatenate([
            np.random.rand(NUM_SHARED_PARAMS), 
            np.ones(num_genes * num_samples), 
            2 * np.ones(num_samples)
        ])
        multi_bounds = [(-15, 15)] * (NUM_SHARED_PARAMS - 1) + \
            [(-np.inf, 1)] + \
            [(-np.inf, np.inf)] * (num_genes * num_samples + num_samples)

        constraints = multi_sample_constraints(num_genes, num_samples)
        custom_err_func = lambda p: multi_sample_error(p, data, num_genes, num_samples)

        optimizer = OptimizerWithEarlyStopping(tol=tol, threshold=threshold)
        res_params, res_err = run_minimize_with_stopping(
            optimizer, 
            initial_params, 
            custom_err_func, 
            constraints = constraints,
            bounds = multi_bounds
        )
        param_res.append((res_err, res_params))
        
    min_param = min(param_res, key = lambda x: x[0])
        
    return min_param[1]

def fit_individual_sample(
    data: list,
    shared_params: list,
    num_iter: int = 1,
    tol: float = 0.5,
    threshold: float = 5,
) -> list:
    assert len(data) > 0

    print("Fitting individual sample...")
    
    # Infer from data
    num_genes = len(data)
    print("Number of genes:", num_genes)
    
    # Collect parameters and errors
    param_res = []
    
    for _ in range(num_iter):
        initial_params = np.concatenate([np.random.rand(num_genes), 2 * np.ones(1)])
        # Generally unconstrained values for DNA and DNA_tot
        bounds = [(-25, 25)] * len(initial_params)
        
        groups = {"G1": list(range(1, num_genes + 1))}
        constraints = single_sample_constraints(groups)
        custom_err_func = lambda p: single_sample_error(p, shared_params, data, num_genes)

        optimizer = OptimizerWithEarlyStopping(tol=tol, threshold=threshold)
        res_params, res_err = run_minimize_with_stopping(
            optimizer, 
            initial_params, 
            custom_err_func, 
            constraints = constraints,
            bounds = bounds
        )
        
        param_res.append((res_err, res_params))
        
    min_param = min(param_res, key = lambda x: x[0])
    
    return min_param[1]

def fit_all_samples(
    # Normalized data
    data: list,
    # Result from fit_shared_params
    shared_params: list,
    num_iter: int = 1,
    tol: float = 0.5,
    threshold: float = 5,
) -> list:
    """
    Manages parallelization of individual sample fitting.
    Returns a list of all parameters for every sample / gene pair.
    Shape of final list is: (num_samples, num_genes + 1).
    Returns DNA concentrations + DNA_tot at the end of each sample list.
    """
    assert len(data) > 0 and len(data[0]) > 0

    print("Data length:", len(data))
    
    results = [r for r in 
        tqdm(
            Parallel(return_as="generator", n_jobs=-1)(
                delayed(fit_individual_sample)(sample_data, shared_params, tol=tol, threshold=threshold, num_iter=num_iter)
                for sample_data in data
            ),
            total=len(data)
        )
    ]
    
    return results

def get_mses(
    # Normalized data
    data: list,
    shared_params: list,
    # Result from fit_all_samples()
    param_list: list,
) -> list:
    """
    Calculates MSE values for each sample-gene pair. Returns a list of lists.
    Shape: (num_samples, num_genes)
    """
    assert len(data) > 0 and len(data[0]) > 0
    
    # Infer from data
    num_samples = len(data)
    num_genes = len(data[0])
    num_timesteps = len(data[0][0])
    
    theta_list = [
        [shared_params + [sample_params[-1], sample_params[gene]] 
         for gene in range(len(data[0]))] for sample_params in param_list
    ]
    
    flat_theta_list = [indiv_theta for sample_thetas in theta_list for indiv_theta in sample_thetas]
    
    results = [r for r in 
        tqdm(
            Parallel(return_as="generator", n_jobs=-1)(
                delayed(lambda params: y_model(params, steps=num_timesteps))(theta)
                for theta in flat_theta_list
            ),
            total=len(flat_theta_list)
        )
    ]
    
    # Calculate MSE per theta
    flat_data = [indiv_data for sample_data in data for indiv_data in sample_data]
    flat_mses = [np.sum((flat_data[i] - results[i])**2) for i in range(len(results))]
    
    return [[flat_mses[sample * num_genes + gene] for gene in range(num_genes)] for sample in range(num_samples)]

def get_end_values(
    data: list,
) -> list:
    """
    Calculates end fluorescence values. Can use this to determine proximity
    to limit of detection.
    """
    assert len(data) > 0 and len(data[0]) > 0
    
    # Infer from data
    num_samples = len(data)
    num_genes = len(data[0])
    
    return [[data[sample][gene][-1] for gene in range(num_genes)] for sample in range(num_samples)]