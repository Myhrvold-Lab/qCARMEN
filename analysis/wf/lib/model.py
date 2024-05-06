from typing import Tuple, Optional, Callable
import numpy as np
from scipy import integrate

def simulation(
    z0: list,
    t_init: float, 
    t_max: float, 
    dt_tols: list, 
    # Differential equation parameters
    params: list,
    # Function that returns differential equation functions
    get_diff_eqs: Callable,
    t_eval: Optional[np.array] = None
) -> Tuple[np.array, np.array]:
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

def eqs_log(log_params: np.array) -> Tuple[float, float, float, float]:
    # Converts from log space to normal space
    params = np.exp(log_params)
    
    # Unpackage parameters
    kcat_trans, kcat_cis, kcat_t7, kf_cis_target, kr_cis_target, \
        kf_cis_other, kr_cis_other, kf_trans, kr_trans, Rep_0, DNA_tot, DNA = params
    
    # Clip DNA if DNA > DNA_tot
    if DNA >= DNA_tot: DNA = DNA_tot - 1e-30

    # Set known reagent concentration parameters
    E_0 = Rep_0 / 50
    T7_0 = Rep_0 * 7.2 / 500
    
    # Fluorescent reporter (separated from quencher)
    def dF_dt(variables):
        F, AE, T_t, T_o = variables
        
        # Uncleaved reporter = initial reporter conc. - cleaved reporter conc.
        FR = Rep_0 - F
        
        return kcat_trans * AE * FR
    
    # Active Cas13 enzyme
    def dAE_dt(variables):
        F, AE, T_t, T_o = variables
        
        # Uncleaved reporter = initial reporter conc. - cleaved reporter conc.
        FR = Rep_0 - F
        c13 = (E_0 - AE - kf_trans * AE * (T_t + T_o + FR) / (kcat_trans + kr_trans)) / (1 + kf_cis_target * T_t / (kr_cis_target + kcat_cis) + kf_cis_other * T_o / kr_cis_other)

        c13_target = kf_cis_target * c13 * T_t / (kr_cis_target + kcat_cis)
        
        return kcat_cis * c13_target
    
    # Transcribed target
    def dTt_dt(variables):
        F, AE, T_t, T_o = variables

        return kcat_t7 * DNA * T7_0
        
    def dTo_dt(variables):
        F, AE, T_t, T_o = variables

        return kcat_t7 * (DNA_tot - DNA) * T7_0
    
    return(dF_dt, dAE_dt, dTt_dt, dTo_dt)

def y_model(theta: list, t_max: float = 1, steps: int = 100) -> np.array:
    _, z_t = simulation(
        [0, 0, 0, 0], 0, t_max, [1e-6, 1e-9], 
        theta, eqs_log, t_eval=np.arange(0, t_max, t_max / steps) + t_max / steps)
    
    return z_t[0, :]