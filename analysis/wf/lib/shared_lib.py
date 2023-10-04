"""
Helper functions for raveling and unraveling arrays for shared fitting.
"""

def get_params(
    # True/False for each parameter if shared or not
    shared_bool,
    # Unraveled list of parameters for shared fitting
    theta,
    # Total number of dilutions theta is for
    num_dil
):
    """
    Returns a list of n lists, where n is the number of unique model parameters. Each 
    sublist contains the values for that model parameter across dilutions.
    """
    # First, calculate number of fixed params
    num_fixed = len(shared_bool) - sum(shared_bool)
    
    # Separate theta into fixed params and shared
    fixed_list = list(theta[:num_fixed])
    shared_list = list(theta[num_fixed:])
    
    # Initialize array for all params
    all = []
    shared_ind = 0
    for shared in shared_bool:
        if not shared: all.append([fixed_list.pop(0)])
        if shared:
            all.append([shared_list[shared_ind * num_dil + iteration] for iteration in range(num_dil)])
            shared_ind += 1
    
    return all

# Takes a "shared" theta array and returns an individual theta given the iteration
def unravel_theta(shared_bool, current_theta, num_dil, iteration):
    """
    Takes a list of parameters used for shared fitting and returns 
    """
    # First, calculate number of fixed params
    num_fixed = len(shared_bool) - sum(shared_bool)
    
    # Separate theta into fixed params and shared
    fixed_list = list(current_theta[:num_fixed])
    shared_list = list(current_theta[num_fixed:])
    
    # Initialize new theta
    new_theta = []
    
    # Loop through boolean array
    shared_ind = 0
    for shared in shared_bool:
        # If parameter is fixed, pop it from fixed and add to new_theta
        if not shared: new_theta.append(fixed_list.pop(0))
        # Otherwise, we're going to want to get the index of the dataset of interest
        if shared:
            new_theta.append(shared_list[shared_ind * num_dil + iteration])
            shared_ind += 1
    
    return new_theta

# Returns a shared theta array based on boolean and initial guesses
def shared_theta(
    shared_bool,
    init_theta,
    num_dil
):
    # Shared bool will look something like: [False, True, False, False]
    fixed_list = []
    shared_list = []
    
    # Loop through the shared true/false array
    for ind, shared in enumerate(shared_bool):
        # Append to fixed if not shared
        if not shared: fixed_list.append(init_theta[ind])
        # Append num_dil times to shared if it is shared
        if shared: [shared_list.append(init_theta[ind]) for x in range(num_dil)]
    
    # The shared theta array will have the non-shared first then the shared
    return fixed_list + shared_list