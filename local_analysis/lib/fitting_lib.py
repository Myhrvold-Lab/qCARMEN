import numpy as np
from joblib import Parallel, delayed

from .constants import NUM_SHARED_PARAMS
from .model import y_model


def multi_sample_model(
    params: list,
    num_genes: int,
    num_samples: int,
    num_timesteps: int = 37,
) -> list:
    num_shared = NUM_SHARED_PARAMS
    shared_params = params[:num_shared]
    dna_vals = params[num_shared : num_shared + num_samples * num_genes]
    dna_tot_vals = params[num_shared + num_samples * num_genes :]

    job_items = []
    for i in range(num_samples):
        for j in range(num_genes):
            dna = dna_vals[i * num_genes + j]
            dna_tot = dna_tot_vals[i]
            theta = shared_params.tolist() + [dna_tot, dna]
            job_items.append(theta)

    results = [
        r
        for r in Parallel(return_as="generator", n_jobs=-1)(
            delayed(lambda params: y_model(params, steps=num_timesteps))(theta)
            for theta in job_items
        )
    ]
    all_results = [
        [results[i * num_genes + j] for j in range(num_genes)]
        for i in range(num_samples)
    ]

    return all_results


def multi_sample_constraints(num_genes: int, num_samples: int) -> list:
    """
    Returns constraints for initial multi-sample fitting process.
    """
    constraints = []
    start = NUM_SHARED_PARAMS

    # Calculate indices for DNA and DNA_tot
    # DNA indices for each gene in each sample
    dna_indices = [start + i for i in range(num_samples * num_genes)]
    # DNA_tot for each sample is located after all DNA values for that sample
    dna_tot_indices = [start + num_samples * num_genes + i for i in range(num_samples)]

    for i in range(num_samples):
        # DNA indices for i-th sample
        sample_dna_indices = dna_indices[i * num_genes : (i + 1) * num_genes]
        # DNA_tot index for i-th sample
        dna_tot_index = dna_tot_indices[i]

        # Constraint: sum of DNAs equals DNA_tot for the sample
        constraints.append(
            {
                "type": "eq",
                "fun": lambda p,
                sample_indices=sample_dna_indices,
                tot_index=dna_tot_index: sum(np.exp(p[idx]) for idx in sample_indices)
                - np.exp(p[tot_index]),
            }
        )

    return constraints


def single_sample_model(
    fixed_params: list,
    variable_params: list,
    num_genes: int,
    num_timesteps: int = 37,
) -> list:
    job_items = []
    for i in range(num_genes):
        dna = variable_params[i]
        dna_tot = variable_params[-1]
        theta = fixed_params.tolist() + [dna_tot, dna]
        job_items.append(theta)

    results = []
    for theta in job_items:
        results.append(y_model(theta, steps=num_timesteps))

    return results


def single_sample_constraints(groups: list) -> list:
    """
    Groups is a dict of lists. Each list contains 1-indexed indices
    of assay wells that correspond to a single group / replicate of
    genes e.g. repeated crRNAs in assay wells due to fill otherwise
    empty space on chip.
    """
    constraints = []
    for item in groups:
        group_inds = np.array(groups[item]) - 1
        print("Group inds:", group_inds)
        constraints.append(
            {
                "type": "eq",
                "fun": lambda p: np.sum(np.exp(p)[group_inds]) - np.exp(p[-1]),
            }
        )

    return constraints


def multi_sample_error(
    params: list, y_data: list, num_genes: int, num_samples: int
) -> float:
    num_timesteps = len(y_data[0][0])
    model_outputs = multi_sample_model(
        params, num_genes, num_samples, num_timesteps=num_timesteps
    )
    error = 0
    for i in range(num_samples):
        for j in range(num_genes):
            error += np.sum((y_data[i][j] - model_outputs[i][j]) ** 2)

    return error


def single_sample_error(
    params: list, fixed_params: list, y_data: list, num_genes: int
) -> float:
    num_timesteps = len(y_data[0])
    model_outputs = single_sample_model(
        fixed_params, params, num_genes, num_timesteps=num_timesteps
    )
    error = 0
    for j in range(num_genes):
        error += np.sum((y_data[j] - model_outputs[j]) ** 2)

    return error


def normalize_data(data: list, norm_max: float = 0.95) -> list:
    """
    Input: list of lists. Should have 192 (for each sample) sublists
    with 24 (for each gene) time series sets in each sublist.

    Output: list of lists with same shape.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    # Subtract min value
    new_data = [
        [(d - min_val) / (max_val - min_val) * norm_max for d in sample_arr]
        for sample_arr in data
    ]

    return new_data


def select_representative_samples(data: list, num_reps: int = 2) -> list[int]:
    assert num_reps > 1, "num_reps must be greater than 1."
    end_totals = np.array([np.sum([p[-1] for p in s]) for s in data])

    # Indices for the largest and smallest values
    max_ind = int(np.argmax(end_totals))
    min_ind = int(np.argmin(end_totals))

    # Calculate halfway value
    midpoint = (end_totals[max_ind] + end_totals[min_ind]) / 2

    # Finding the index of the value closest to halfway
    mid_ind = int(np.argmin(np.abs(end_totals - midpoint)))

    if num_reps == 2:
        return [max_ind, mid_ind]

    # Otherwise, return smallest to highest, equally spaced
    k = min(num_reps, end_totals.size)
    order = np.argsort(end_totals)
    pos = np.linspace(0, end_totals.size - 1, k).round().astype(int)
    chosen = order[pos].tolist()
    return chosen
