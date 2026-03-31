from typing import Tuple, Optional, Callable
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc


def run_least_squares(
    residual_func: Callable,
    x0: np.ndarray,
    bounds: list,
    method: str = "trf",
    max_nfev: int = 5000,
) -> Tuple[np.array, float]:
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    result = least_squares(
        residual_func,
        x0,
        bounds=(lower, upper),
        method=method,
        max_nfev=max_nfev,
    )

    return result.x, result.cost


def generate_initial_params_lhs(
    bounds: list,
    n_samples: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    n_params = len(bounds)
    sampler = qmc.LatinHypercube(d=n_params, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    return qmc.scale(unit_samples, lower, upper)
