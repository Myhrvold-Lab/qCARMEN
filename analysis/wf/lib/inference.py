"""
All functions related to actually calculating concentrations.
"""
import numpy as np

from .model_fitting import get_fixed_fit, get_shared_fit
from .shared_lib import unravel_theta

def train_gene_model(
    # Biomark data object
    data,
    # ID of the gene we're training on
    gene_id,
    # Well IDs to train on
    train_wells,
    # Initial guess for model parameters
    init_guess=[0.8218, 0.2063, 1.252, 1.112, 1.358, -0.3494, -0.7695, -2.748, 0.6502, -0.3509, 0],
    # Initial concentration guess
    conc_guess=None,
    # Correction factor for FAM/ROX values
    correction_factor=0,
):
    """
    Gets parameters for a gene in the given sample set.
    """
    bools = [
        False, False, False, False, False, False,
        False, False, False, False, True
    ]

    train_data = [data.get_fam_rox(train_well, gene_id) - correction_factor for train_well in train_wells]
    
    # If an init conc. guess is not supplied, use an arbitrary set of params to get conc values
    if conc_guess is None:
        conc_guess = []
        for train_vals in train_data:
            fixed_res = get_fixed_fit(init_guess, train_vals)
            conc_guess.append(fixed_res[0])
    
    # Now fit on the range of datapoints we get
    goi_shared_thetas = get_shared_fit(train_data, conc_guess)
    
    return unravel_theta(bools, goi_shared_thetas, len(train_data), 1)

def get_optimal_train_set(
    # Biomark data object
    data,
    # ID of the gene we're training on
    gene_id,
    # Wells that are available for testing (all as default)
    train_cands = [x + 1 for x in range(192)],
    # Initial guess for model parameters
    init_guess=[0.8218, 0.2063, 1.252, 1.112, 1.358, -0.3494, -0.7695, -2.748, 0.6502, -0.3509, 0],
):
    """
    Returns a list of four well IDs that capture a wide range of concentrations as
    well as concentration guesses for thos wells.
    """
    # First, loop through all wells and get their concentration values
    init_all_concs = get_concentrations(data, gene_id, train_cands, init_guess)

    # Order wells by concentration
    sort_concs = np.array(init_all_concs)

    # Get min/max concs
    q1 = sort_concs[0]
    q4 = sort_concs[-1]

    # Get inds of quartiles
    q2 = (np.abs(sort_concs - (1 * (q4 - q1) / 3 + q1))).argmin()
    q3 = (np.abs(sort_concs - (2 * (q4 - q1) / 3 + q1))).argmin()

    sort_inds = np.argsort(sort_concs)

    return [sort_inds[0], sort_inds[q2], sort_inds[q3], sort_inds[-1]]

def get_concentrations(
    # Biomark data object
    data,
    # ID of the gene we're training on
    gene_id,
    # Wells to calculate concentrations for
    wells,
    # Parameters to use for this gene
    gene_params,
):
    """
    Calculates concentrations for all wells supplied using supplied parameters for that 
    gene, most likely calculated from train_gene_model().
    """
    res = []
    for well in wells:
        fixed_data = data.get_fam_rox(well, gene_id)
        fixed_res = get_fixed_fit(gene_params, fixed_data)
        res.append(fixed_res[0])

    return res

def get_representative_wells(
    # Biomark data object
    data,
    goi_gene_id,
    # Wells to search (default: all 192 wells in order)
    wells=[x for x in range(1, 193)],
    # Initial guess for theta, should be 11 values
    init_theta=[0.8218,0.2063,1.252,1.112,1.358,-0.3494,-0.7695,-2.748,0.6502,-0.3509,0],
    # How many representative wells to return and how spread out they should be
    split=[0, 0.5, 1],
):
    """
    Returns a list of wells that give a good spread of training values for the model. Looking for 
    a really high value, a really low value, and a medium value.

    It's not sufficient to sort and choose the 0, 96, and 191st wells because a large portion of 
    wells might be empty or extremely saturated.

    We provide a parameter called "split" that contains the ideal concentration ranges we want. For 
    example, if we have split=[0, 0.5, 1], then we want to return three wells that have calculated 
    concentrations that are the minimum, 50th percentile, and maximum of all wells. If we have 
    split=[0, 0.3, 0.9, 1], then we want to return four wells that have calculated concentrations 
    that are the minimum, 30th percentile, 90th percentile, and maximum of all wells.
    """

    init_goi_indiv = []
    # Loop through every well
    for well in wells:
        fixed_data = data.get_fam_rox(well, goi_gene_id)
        # Actually calculate that model fit for that data
        fixed_res = get_fixed_fit(init_theta, fixed_data)
        
        init_goi_indiv.append(fixed_res[0])

    # Sort init_goi_indiv and keep track of the indices in a separate array
    sorted_inds = np.argsort(init_goi_indiv)
    sorted_vals = np.sort(init_goi_indiv)

    # Renormalize split range based on sorted min/max
    split_normed = [sorted_vals[0] + (sorted_vals[-1] - sorted_vals[0]) * x for x in split]

    # Get the indices of sorted_vals that are closest to each split_normed value
    split_inds = []
    for split_val in split_normed:
        split_inds.append(sorted_inds[np.abs(sorted_vals - split_val).argmin()])

    # Return the wells themselves
    return [wells[ind] for ind in split_inds]