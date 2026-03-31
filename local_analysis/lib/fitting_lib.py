import numpy as np
from scipy.special import logsumexp
from joblib import Parallel, delayed

from .constants import NUM_SHARED_PARAMS
from .model import y_model


def _run_y_model(theta, steps):
    return y_model(theta, steps=steps)


def compute_dna_tot(dna_log_values: np.ndarray) -> float:
    return logsumexp(dna_log_values)


def multi_sample_model(
    params: list,
    num_genes: int,
    num_samples: int,
    num_timesteps: int = 37,
    n_jobs: int = 1,
) -> list:
    num_shared = NUM_SHARED_PARAMS
    shared_params = params[:num_shared]
    dna_vals = params[num_shared : num_shared + num_samples * num_genes]

    job_items = []
    for i in range(num_samples):
        sample_dna = dna_vals[i * num_genes : (i + 1) * num_genes]
        dna_tot = compute_dna_tot(sample_dna)
        for j in range(num_genes):
            dna = sample_dna[j]
            theta = shared_params.tolist() + [dna_tot, dna]
            job_items.append(theta)

    if n_jobs == 1:
        results = [y_model(theta, steps=num_timesteps) for theta in job_items]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_y_model)(theta, num_timesteps) for theta in job_items
        )

    all_results = [
        [results[i * num_genes + j] for j in range(num_genes)]
        for i in range(num_samples)
    ]

    return all_results


def single_sample_model(
    fixed_params: list,
    variable_params: list,
    num_genes: int,
    num_timesteps: int = 37,
) -> list:
    dna_vals = variable_params[:num_genes]
    dna_tot = compute_dna_tot(np.array(dna_vals))

    job_items = []
    for i in range(num_genes):
        dna = dna_vals[i]
        theta = fixed_params.tolist() + [dna_tot, dna]
        job_items.append(theta)

    results = []
    for theta in job_items:
        results.append(y_model(theta, steps=num_timesteps))

    return results


def multi_sample_residuals(
    params: list, y_data: list, num_genes: int, num_samples: int,
    n_jobs: int = 1, reg_lambda: float = 0.0,
) -> np.ndarray:
    num_timesteps = len(y_data[0][0])
    model_outputs = multi_sample_model(
        params, num_genes, num_samples, num_timesteps=num_timesteps, n_jobs=n_jobs,
    )
    residuals = []
    for i in range(num_samples):
        for j in range(num_genes):
            r = y_data[i][j] - model_outputs[i][j]
            residuals.append(r)

    flat = np.concatenate(residuals)
    if reg_lambda > 0:
        dna_params = np.array(params[NUM_SHARED_PARAMS:])
        flat = np.concatenate([flat, np.sqrt(reg_lambda) * dna_params])

    return flat


def single_sample_residuals(
    params: list, fixed_params: list, y_data: list, num_genes: int,
    reg_lambda: float = 0.0,
) -> np.ndarray:
    """Return flat residual vector (y_data - model) for least_squares."""
    num_timesteps = len(y_data[0])
    model_outputs = single_sample_model(
        fixed_params, params, num_genes, num_timesteps=num_timesteps
    )
    residuals = []
    for j in range(num_genes):
        r = y_data[j] - model_outputs[j]
        residuals.append(r)

    flat = np.concatenate(residuals)
    if reg_lambda > 0:
        flat = np.concatenate([flat, np.sqrt(reg_lambda) * np.array(params[:num_genes])])

    return flat


def normalize_data(data: list, norm_max: float = 0.95) -> list:
    """Normalize fluorescence data to [0, norm_max] range."""
    min_val = np.min(data)
    max_val = np.max(data)
    new_data = [
        [(d - min_val) / (max_val - min_val) * norm_max for d in sample_arr]
        for sample_arr in data
    ]
    return new_data


def select_representative_samples(data: list, num_reps: int = 2) -> list[int]:
    """Select representative samples spanning the range of endpoint fluorescence."""
    assert num_reps > 1, "num_reps must be greater than 1."
    end_totals = np.array([np.sum([p[-1] for p in s]) for s in data])

    max_ind = int(np.argmax(end_totals))
    min_ind = int(np.argmin(end_totals))
    midpoint = (end_totals[max_ind] + end_totals[min_ind]) / 2
    mid_ind = int(np.argmin(np.abs(end_totals - midpoint)))

    if num_reps == 2:
        return [max_ind, mid_ind]

    k = min(num_reps, end_totals.size)
    order = np.argsort(end_totals)
    pos = np.linspace(0, end_totals.size - 1, k).round().astype(int)
    chosen = order[pos].tolist()

    return chosen
