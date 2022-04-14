from scipy import interpolate, integrate

# Function to cut out islands: only keep the turn number if smaller than the previous one.
# Otherwise, replace by previous turn number.
descend = np.frompyfunc(lambda x, y: y if y < x else x, 2, 1)

def _calculate_radial_evo(data):
    # We select the window in number of turns for which each angle has values
    minturns = np.array([ x[1].min() for x in data.values()]).max()
    maxturns = np.array([ x[1].max() for x in data.values()]).min()
    turns = np.unique([ x for x in np.concatenate([ x[1] for x in data.values()]) if (x >= minturns and x <= maxturns) ])
    angles=np.array(list(data.keys()))
    # initialise numpy array
    result = {turn: np.zeros(len(angles)) for turn in turns }
    for i_ang,ang in enumerate(angles):
        rho_surv = np.flip(data[ang])
        # Shift the value of Nturn slightly up such that, in case of a vertical cliff,
        # the interpolation will assign the largest rho_surv value. This cannot be
        # done for the smallest Nturn as this will give an out-of-bounds error.
        rho_surv[0,1:] += 1e-7
        rho_turns = interpolate.interp1d(rho_surv[0], rho_surv[1], kind='linear')(turns)
        for i_turn, turn in enumerate(turns):
            result[turn][i_ang] = rho_turns[i_turn]
    da = np.zeros(len(turns))
    for i_turn, turn in enumerate(turns):
        da[i_turn] = np.sqrt(
            2/np.pi*integrate.trapezoid(x=angles*np.pi/180, y=result[turn]**2)
        )
    return np.array([turns, da])
    
def _get_raw_da_radial(data):
    angles = np.unique(data.angle)
    ampcol = 'amplitude'
    da = {}
    for angle in angles:
        datasort = data[data.angle==angle].sort_values(ampcol)
        # Cut out islands from evolution (i.e. force non-monotonously descending)
        datasort['turns_no_islands'] = descend.accumulate(datasort.turns.astype(object))
        # Keep the values around the places where the number of turns changes
        mask = (datasort.turns_no_islands.diff() < 0 ) | (datasort.turns_no_islands.shift(-1).diff() < 0)
        mask.iloc[-1] = True  # Add the last amplitude step (needed for interpolation later)
        da[angle] = np.array((datasort[mask][ampcol].values,datasort[mask].turns_no_islands.values), dtype=float)
    return da

def get_da_radial(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return np.array([ [ang, vals[0,0]] for ang, vals in _get_raw_da_radial(data).items() ])

def get_da_evo_radial(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return _calculate_radial_evo(_get_raw_da_radial(data))
    


def _get_raw_da_sixdesk(data):
    angles = np.unique(data.angle)
    seeds = np.unique(data.seed)
    ampcol = 'amp'
    da = {}
    data['turns'] = data[['sturns1']] #,'sturns2']].min(axis=1)
    for seed in seeds:
        da[seed] = {}
        for angle in angles:
            datasort = data[(data.angle==angle) & (data.seed==seed)].sort_values(ampcol)
            # Cut out islands from evolution (i.e. force non-monotonously descending)
            datasort['turns_no_islands'] = descend.accumulate(datasort.turns.astype(object))
            # Keep the values around the places where the number of turns changes
            mask = (datasort.turns_no_islands.diff() < 0 ) | (datasort.turns_no_islands.shift(-1).diff() < 0)
            mask.iloc[-1] = True  # Add the last amplitude step (needed for interpolation later)
            da[seed][angle] = np.array((
                                datasort[mask][ampcol].values,datasort[mask].turns_no_islands.values
                            ), dtype=float)
    return da

def get_da_sixdesk(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return {
        seed: np.array([ [ang, vals[0,0]] for ang, vals in da.items() ])
        for seed, da in _get_raw_da_sixdesk(data).items()
    }

def get_da_evo_sixdesk(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return { seed: _calculate_radial_evo(da) for seed,da in _get_raw_da_sixdesk(data).items() }