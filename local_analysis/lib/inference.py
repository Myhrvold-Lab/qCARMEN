import numpy as np
from typing import Optional
from tqdm import tqdm
from joblib import Parallel, delayed

from .model import y_model
from .optimizer_lib import (
    run_least_squares,
    generate_initial_params_lhs,
)
from .fitting_lib import (
    compute_dna_tot,
    multi_sample_residuals,
    single_sample_residuals,
)
from .constants import NUM_SHARED_PARAMS, SHARED_PARAM_BOUNDS, DNA_BOUNDS


def fit_shared_params(
    data: list,
    num_iter: int = 16,
    seed: Optional[int] = None,
    lsq_method: str = "trf",
    reg_lambda: float = 0.01,
) -> list:
    assert len(data) > 0

    num_samples = len(data)
    num_genes = len(data[0])

    num_dna_params = num_genes * num_samples
    bounds = list(SHARED_PARAM_BOUNDS) + [DNA_BOUNDS] * num_dna_params

    raw_resid_func = lambda p: multi_sample_residuals(
        p, data, num_genes, num_samples,
        n_jobs=-1, reg_lambda=reg_lambda,
    )

    if reg_lambda > 0:
        print(f"Regularization: L2 lambda={reg_lambda}")

    param_res = []
    initial_params_set = generate_initial_params_lhs(bounds, num_iter, seed=seed)

    for iter_ind in range(num_iter):
        try:
            print(f"Beginning least_squares ({lsq_method}) iteration {iter_ind + 1} of {num_iter}...")
            res_params, res_cost = run_least_squares(
                raw_resid_func,
                initial_params_set[iter_ind],
                bounds,
                method=lsq_method,
            )
            param_res.append((res_cost, res_params))
        except Exception as e:
            print(f"least_squares failed: {e}")

    min_param = min(param_res, key=lambda x: x[0])
    print(f"least_squares best cost: {min_param[0]:.6f}")
    return min_param[1]


def fit_individual_sample(
    data: list,
    shared_params: list,
    num_iter: int = 3,
    initial_dna: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    lsq_method: str = "trf",
    reg_lambda: float = 0.0,
) -> list:
    assert len(data) > 0

    print("Fitting individual sample...")

    num_genes = len(data)
    print("Number of genes:", num_genes)

    bounds = [DNA_BOUNDS] * num_genes

    resid_func = lambda p: single_sample_residuals(
        p, shared_params, data, num_genes, reg_lambda=reg_lambda,
    )

    param_res = []

    for iter_idx in range(num_iter):
        try:
            if iter_idx == 0 and initial_dna is not None:
                init_params = np.array(initial_dna[:num_genes], dtype=float)
            else:
                init_set = generate_initial_params_lhs(bounds, 1, seed=(seed + iter_idx) if seed is not None else None)
                init_params = init_set[0]

            res_params, res_err = run_least_squares(
                resid_func, init_params, bounds, method=lsq_method,
            )

            param_res.append((res_err, res_params))
        except:
            print("inf or nan generated in individual fitting.")

    if not param_res:
        print("WARNING: all restarts failed for sample, returning zeros")
        return np.zeros(num_genes)

    min_param = min(param_res, key=lambda x: x[0])

    return min_param[1]


def fit_all_samples(
    data: list,
    shared_params: list,
    num_iter: int = 3,
    stage1_dna: Optional[np.ndarray] = None,
    rep_indices: Optional[list] = None,
    seed: Optional[int] = None,
    lsq_method: str = "trf",
    reg_lambda: float = 0.0,
) -> list:
    assert len(data) > 0 and len(data[0]) > 0

    num_samples = len(data)
    num_genes = len(data[0])

    print("Data length:", num_samples)

    # Build warm-start DNA values for each sample
    warm_starts = [None] * num_samples

    if stage1_dna is not None and rep_indices is not None:
        # Warm-start representatives directly from Stage 1 DNA values
        for i, rep_idx in enumerate(rep_indices):
            if rep_idx < num_samples:
                start = i * num_genes
                end = start + num_genes
                if end <= len(stage1_dna):
                    warm_starts[rep_idx] = stage1_dna[start:end]

        # Warm-start non-representatives by scaling reference DNA by endpoint fluorescence ratio
        ref_idx = None
        ref_dna = None
        for i, rep_idx in enumerate(rep_indices):
            if warm_starts[rep_idx] is not None:
                ref_idx = rep_idx
                ref_dna = warm_starts[rep_idx]
                break

        if ref_dna is not None:
            ref_end_vals = np.array([data[ref_idx][g][-1] for g in range(num_genes)])
            ref_end_vals = np.clip(ref_end_vals, 1e-10, None)

            for s in range(num_samples):
                if warm_starts[s] is None:
                    sample_end_vals = np.array([data[s][g][-1] for g in range(num_genes)])
                    sample_end_vals = np.clip(sample_end_vals, 1e-10, None)
                    ratio = np.log(sample_end_vals / ref_end_vals)
                    warm_starts[s] = ref_dna + ratio

    results = [r for r in
        tqdm(
            Parallel(return_as="generator", n_jobs=-1)(
                delayed(fit_individual_sample)(
                    sample_data, shared_params,
                    num_iter=num_iter,
                    initial_dna=warm_starts[i],
                    seed=seed,
                    lsq_method=lsq_method,
                    reg_lambda=reg_lambda,
                )
                for i, sample_data in enumerate(data)
            ),
            total=num_samples,
        )
    ]

    return results


def get_mses(
    data: list,
    shared_params: list,
    param_list: list,
) -> list:
    assert len(data) > 0 and len(data[0]) > 0

    num_samples = len(data)
    num_genes = len(data[0])
    num_timesteps = len(data[0][0])

    theta_list = []
    for sample_params in param_list:
        dna_vals = np.array(sample_params[:num_genes])
        dna_tot = compute_dna_tot(dna_vals)
        sample_thetas = [
            shared_params + [dna_tot, sample_params[gene]]
            for gene in range(num_genes)
        ]
        theta_list.append(sample_thetas)

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

    flat_data = [indiv_data for sample_data in data for indiv_data in sample_data]
    flat_mses = [np.sum((flat_data[i] - results[i])**2) for i in range(len(results))]

    return [[flat_mses[sample * num_genes + gene] for gene in range(num_genes)] for sample in range(num_samples)]


def get_end_values(data: list) -> list:
    """Calculates end fluorescence values for limit-of-detection assessment."""
    assert len(data) > 0 and len(data[0]) > 0

    num_samples = len(data)
    num_genes = len(data[0])

    return [[data[sample][gene][-1] for gene in range(num_genes)] for sample in range(num_samples)]
