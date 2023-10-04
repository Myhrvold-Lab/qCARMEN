import numpy as np
import scipy
from scipy import integrate
from scipy.optimize import least_squares

from .shared_lib import shared_theta, unravel_theta

def simulation(
    z0, 
    t_init, 
    t_max, 
    dt_tols, 
    # Differential equation parameters
    params,
    # Function that returns differential equation functions
    get_diff_eqs,
    t_eval=None
):
    """
    Solves the differential equations from t_init to t_max and with initial conditions z0.
    """
    # Unpackage differential equations from the get_diff_eqs function
    eqs_res = get_diff_eqs(params)
    def dz_dt(variables):
        return(np.array([eq(variables) for eq in eqs_res]))
    
    def diff_eqs_sol(tSpan, params, z0):
        # solve ODE
        odeSol = integrate.solve_ivp(
            lambda tSpan, z: dz_dt(z),
            tSpan, 
            z0, 
            t_eval=t_eval,
            method = 'Radau',
            vectorized=False,
            rtol=dt_tols[0], atol=dt_tols[1],  # default 1e-3 and 1e-6
        )

        z = odeSol.y
        t = odeSol.t

        return t, z

    t_vec, z_t = diff_eqs_sol([t_init, t_max], [], z0)
    
    return(t_vec, z_t)

def eqs_log(log_params):
    # Converts from log space to normal space
    params = np.power(10, log_params)
    
    # Unpackage parameters
    kcat_trans, kcat_cis, kcat_t7, kf_cis, kf_trans, \
        kr_trans, kr_cis, kf_pre, kr_pre, Rep_0, DNA = params
    
    # Set known reagent concentration parameters
    E_0 = Rep_0 / 50
    T7_0 = Rep_0 * 7.2 / 500
    cr_0 = E_0

    # Michaelis constants
    KM_trans = (kr_trans + kcat_trans) / kf_trans
    KM_cis = (kr_cis + kcat_cis) / kf_cis
    
    # Fluorescent reporter (separated from quencher)
    def dF_dt(variables):
        F, AE, T, C13 = variables
        
        # Uncleaved reporter = initial reporter conc. - cleaved reporter conc.
        FR = (Rep_0 - F) / (1 + AE / KM_trans)
        
        return kcat_trans * AE * FR / KM_trans
    
    # Active Cas13 enzyme
    def dAE_dt(variables):
        F, AE, T, C13 = variables
        
        # Uncleaved reporter = initial reporter conc. - cleaved reporter conc.
        FR = (Rep_0 - F) / (1 + AE / KM_trans)
        
        # Cas13-crRNA complex
        c13_cr = pow(T / KM_cis + 1, -1) * (E_0 - AE * (1 + (T + FR) / KM_trans) - C13)
        
        return kcat_cis / KM_cis * c13_cr * T
    
    # Transcribed target
    def dT_dt(variables):
        F, AE, T, C13 = variables
        
        # Uncleaved reporter = initial reporter conc. - cleaved reporter conc.
        FR = (Rep_0 - F) / (1 + AE / KM_trans)
        
        # Cas13-crRNA complex
        c13_cr = pow(T / KM_cis + 1, -1) * (E_0 - AE * (1 + (T + FR) / KM_trans) - C13)
        
        return kcat_t7 * T7_0 * DNA + c13_cr * T * (kr_cis / KM_cis - kf_cis) + AE * T * (kr_trans / KM_trans - kf_trans)
    
    # Free Cas13 enzyme
    def dC13_dt(variables):
        F, AE, T, C13 = variables
        
        # Calculate crRNA concentration from conservation equation
        crRNA = cr_0 + C13 - E_0
        
        # Uncleaved reporter = initial reporter conc. - cleaved reporter conc.
        FR = (Rep_0 - F) / (1 + AE / KM_trans)

        # Cas13-crRNA complex
        c13_cr = pow(T / KM_cis + 1, -1) * (E_0 - AE * (1 + (T + FR) / KM_trans) - C13)
        
        return kr_pre * c13_cr - kf_pre * C13 * crRNA
    
    return(dF_dt, dAE_dt, dT_dt, dC13_dt)

def y_model(theta, x, t_max, steps):
    # Initial Cas13 concentration
    E_0 = np.log10(np.power(10, theta[-2]) / 50)
    
    t_vec, z_t = simulation([0, 0, 0, E_0], 0, t_max, [1e-6, 1e-9], theta, eqs_log, t_eval=np.arange(0, t_max, t_max / steps) + t_max / steps)
    
    # If we don't get a complete result, return negative values
    if len(z_t) == 0 or z_t[0, :].shape[0] < steps: return (-5) * np.ones(steps)
    
    return z_t[0, :]

# Cost function for parameter fitting
def shared_fun(
    theta,
    x_vals,
    y_vals,
    fit_fun, 
    shared_bool,
    num_dil,
):
    # Theta will essentially have the normal parameters that we fix with the variable parameters at the end
    # E.g. [a, b, c, d, e_1, e_2, e_3, e_4]
    # We will have to make "new thetas"
    
    # Initialized calculated y's
    pred_y = []
    
    # Loop through all the different datasets
    for data_ind in range(num_dil):
        new_theta = unravel_theta(shared_bool, theta, num_dil, data_ind)
        predicted_y = fit_fun(new_theta, x_vals, x_vals[-1], x_vals.shape[0])
        pred_y.append(predicted_y)
    
    return np.concatenate(pred_y) - y_vals

# Provided an array of y datasets and a boolean array of parameters that should be shared...
def shared_fit(
    # Time/x values
    x_vals,
    # Datasets we're performing shared fit on
    datasets,
    # What parameters are being shared across dilutions
    shared_bool,
    # Initial guess for model parameters
    init_theta,
    # Initial guess for concentrations of curves in datasets
    init_conc,
    # The function that calls our model
    fit_fun,
    # Lower and upper bounds
    b1, b2,
):
    # datasets example: [[1, 2, 3, 4], [0, 1, 2, 3], [3, 4, 5, 6]]
    # shared_bool example: [False, True, False, False] for a four-parameter model
    
    # We HAVE to put all parameters that are being fit into theta, we can't have that as a separate argument
    init_theta_shared = shared_theta(shared_bool, init_theta, len(datasets))
    
    init_theta_shared[-len(datasets):] = init_conc[:len(datasets)]
    
    # y_vals itself will be a concatenated array
    y_vals = np.concatenate(datasets)
    
    # x_vals can just be rescaled time, going to assume components in datasets
    res = least_squares(shared_fun, init_theta_shared, 
                        ftol=1e-12,
                        gtol=1e-12,
                        xtol=1e-12,
                        max_nfev=1e11,
                        args=(x_vals, y_vals, fit_fun, shared_bool, len(datasets)), 
                        bounds=(b1, b2))
    
    return res

def get_shared_fit(
    datasets,
    init_conc,
):
    # Boolean values
    bools = [
        False, False, False, False, False, False,
        False, False, False, False, True
    ]
    
    # Set initial guess
    initial_guess = unravel_theta(bools, [0.8218, 0.2063, 1.252, 1.112, 1.358, -0.3494, -0.7695, -2.748, 0.6502, -0.3509, 1.190, 0.4226, -0.5118, -1.044, -1.817, -1.947], 6, 2)

    # Calculate bounds
    bounds_std = np.array([10 for x in range(11)])
    lower_bounds = shared_theta(bools, np.array(initial_guess) - 2 * bounds_std, len(datasets))
    upper_bounds = shared_theta(bools, np.array(initial_guess) + 2 * bounds_std, len(datasets))

    # x_model = np.array(range(1, 38)) * 3
    x_predicted = np.array(range(1, 38)) * 100 / 37

    # Perform shared fit with function of interest
    shared_res_default = shared_fit(x_predicted, datasets, bools, initial_guess, init_conc, y_model, lower_bounds, upper_bounds)

    # Otherwise, return default res
    return shared_res_default.x

# Performs a fixed fit for an individual dataset
def fixed_fit(x_vals, dataset, fixed_bool, init_theta, fit_fun):
    # Fixed bool contains True/False values for whether a theta param should stay fixed
    short_theta = [theta_val for ind, theta_val in enumerate(init_theta) if not fixed_bool[ind]]
    
    res = least_squares(indiv_fun, short_theta, 
                        args=(x_vals, dataset, fit_fun, fixed_bool, init_theta), 
                        max_nfev=1e11, 
                        ftol=1e-12, gtol=1e-12, xtol=1e-12, 
                        bounds=(-np.inf, np.inf)
                       )
    return res

# Cost function for individual parameter fitting
def indiv_fun(theta, x_vals, y_vals, y_fun, fixed_bool, full_theta):
    # Copy original theta
    theta_copy = list(theta).copy()
    
    # Re-create theta based on the fixed_bool
    new_theta = []
    for ind, theta_val in enumerate(full_theta):
        if fixed_bool[ind]: new_theta.append(theta_val)
        else: new_theta.append(theta_copy.pop(0))
            
    return y_fun(new_theta, x_vals, x_vals[-1], x_vals.shape[0]) - y_vals

def get_fixed_fit(
    init_guess,
    dataset,
):
    """
    This function will return a fixed fit.
    """
    x_predicted = np.array(range(1, 38)) * 100 / 37

    fixed_vals = [
        True, True, True, True, True,
        True, True, True, True, True, False
    ]

    # Perform shared fit with function of interest
    fixed_res = fixed_fit(
        x_predicted, 
        dataset, 
        fixed_vals, 
        init_guess, 
        y_model,
    )

    return fixed_res.x