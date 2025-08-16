from typing import Tuple, Optional, Callable
import numpy as np
from scipy.optimize import minimize

class EarlyStopping(Exception):
    def __init__(
        self, 
        message: str, 
        last_params: list, 
        last_value: float,
    ):
        super().__init__(message)
        self.last_params = last_params
        self.last_value = last_value

class OptimizerWithEarlyStopping:
    def __init__(
        self, 
        threshold: float = 0, 
        tol: float = 0.2, 
        beta: float = 0.5, 
        history_len: int = 5,
        iter_limit: int = 50,
        verbose: bool = False,
    ):
        # For error smoothing
        self.beta = beta
        self.threshold = threshold
        self.tol = tol
        self.history_len = history_len
        self.smoothed_error = None
        self.error_history = [float('inf')] * history_len
        # Hold the best error we've gotten so far
        self.best_error = float('inf')
        # Hold the best set of parameters we've tested so far
        self.best_param = None
        self.iter_limit = iter_limit
        self.best_ind = 0
        self.current_ind = 0
        self.verbose = verbose

    def objective(
        self, 
        params: list, 
        error_func: Callable
    ) -> float:
        error = error_func(params)

        # Update the smoothed error
        if self.smoothed_error is None:
            self.smoothed_error = error
            self.best_param = params
        else:
            self.smoothed_error = self.beta * self.smoothed_error + (1 - self.beta) * error

        if error < self.best_error:
            self.best_error = error
            self.best_param = params
            self.best_ind = self.current_ind

        return error

    def callback(self, xk: list):
        self.current_ind += 1
        if self.verbose:
            print(f"Current smoothed error and params: {self.smoothed_error}, {xk}", end="\r")
            print("self.tol:", self.tol)
            print("self.smoothed_error:", self.smoothed_error)
            print("self.error_history:", self.error_history)
            print(sum([self.tol > abs(self.smoothed_error - h) for h in self.error_history]), self.error_history)
        
        if self.current_ind - self.best_ind > self.iter_limit:
            if self.verbose: print(f"Stopping early, more than {self.iter_limit} iterations have passed since error improved: {self.smoothed_error} vs. {self.error_history}")
            raise EarlyStopping("Early stopping condition met", self.best_param, self.best_error)

        if sum([self.tol > abs(self.smoothed_error - h) for h in self.error_history]) == self.history_len:
            if self.verbose: print(f"Stopping early, error is not changing: {self.smoothed_error} vs. {self.error_history}")
            raise EarlyStopping("Early stopping condition met", self.best_param, self.best_error)

        # Check if error is below our threshold
        if self.smoothed_error < self.threshold:
            if self.verbose: print(f"Stopping early, error below threshold: {self.smoothed_error}")
            raise EarlyStopping("Early stopping condition met", self.best_param, self.best_error)
            
        # Otherwise, add the current smoothed error to error history
        self.error_history = [self.smoothed_error] + self.error_history[:-1]

def run_minimize_with_stopping(
    optimizer: OptimizerWithEarlyStopping,
    initial_params: list,
    custom_error: Callable,
    constraints: Optional[list] = None,
    bounds: Optional[list] = None,
) -> Tuple[np.array, float]:
    try:
        result = minimize(
            optimizer.objective, 
            initial_params, 
            args=(custom_error),
            constraints=constraints, 
            method="SLSQP", 
            callback=optimizer.callback, 
            # Arbitrarily set ftol to 1e-15 to customize stopping criteria
            options={"maxiter": 1000, "ftol": 1e-15}, 
            bounds=bounds)
        
        return result.x, -1
    except EarlyStopping as e:
        min_err = e
        print("Last known parameters:", min_err.last_params)
        print("Function value at last parameters:", min_err.last_value)

        return min_err.last_params, min_err.last_value