import sys
import xdyna as xd

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 

# This is an example for a DA study with 60 seeds to be ran on a cluster.
# =======================================================================
# The third step is the calculation of the DA, which again cannot be ran
# in parallel.

# We pass the studyname as an argument to the script
study = str(sys.argv[1])
path  = str(sys.argv[2])

# This will load the existing DA based on the meta file with the same name found in the path variable.
# If no path variable is specified, the meta file is assumed to be in the working directory (if this
# is not the case, the script will create a new DA object which is not the intention in this step).
# Because we specified the file containing the xtrack line in Step 1, or calculated it in Step 1bis,
# it will be automatically loaded as well.
DA = xd.DA(name=study, use_files=True, path=path)



# Estimate the DA border at a specific turn
# -------------------------------------------------------------
# The DA estimation can be computed at one specific turn using "calsulate_da". If not given, the DA 
# estimation is done at the last turn of the simulation, i.e. DA.max_turns.
t=DA.max_turns

DA.calculate_da(at_turn=t)


# Estimate DA Turn-by-Turn evolution (default= from_turn=1e3,to_turn=max_turn)
# -------------------------------------------------------------
# The Turn-by-Turn DA estimation is done by "calculate_davsturns". By default, the range of the DA
# estimation is from 1e3 to DA.max_turns but one can change those using from_turn and to_turn
from_t=1e3
to_t=DA.max_turns

DA.calculate_davsturns(from_turn=from_t,to_turn=to_t)

# Using "calculate_davsturns" might result in improving the DA estimation in case of random particle
# distribution as the code will use DA estimation from other turns in order to constrain DA at higher
# turns.


# Plots
# -------------------------------------------------------------
# Buildin routines have been implement in order to easily plot the results.
type_plot="polar"
fig, ax=plt.subplots(2,1,figsize=(10,10));


# There is a routine ploting the particle distribution. The turn at which the particle status must be 
# taken must be given, otherwise the status at the end of the simulation is used. Similarly, the type 
# of plot must be given: either 'cartesian' for x-y plot and 'polar' for ang-amp plot.
DA.plot_particles(ax[0], 
                  at_turn=t,             # Status at this turn
                  type_plot=type_plot,   # Plot type: 'cartesian' or 'polar'
                  csurviving="blue",     # Color for surviving parts.
                  closses="red",         # Color for losses.
                  size_scaling="log",    # Losses dot size: 'log', 'linear' or None 
                  alpha=1)


# This routine plots the DA border in the same format as the previous function. Similarly to 
# "plot_particle", the turn and the type of plot must be given. The routine plots 2 borders 
# representing an upper and a lower estimation of the DA.
DA.plot_da_border(ax[0], 
                  at_turn=t,             # Status at this turn
                  type_plot=type_plot,   # Plot type: 'cartesian' or 'polar'
                  clower="blue",         # Color for lower da estimation.
                  cupper="red",          # Color for upper da estimation.
                  alpha=1, 
                  label="DA")
ax[0].set_title(f'DA at {t}')
ax[0].legend(prop={'size': 15})


# This routine plots the evolution of the DA (minimum, average and maximum values) as a function of turns. 
# Similarly to "calculate_davsturns" the turn range must be given. The routine also plots the values for 
# the upper and lower estimation of the DA.
DA.plot_davsturns_border(ax[1], 
                         from_turn=from_t, 
                         to_turn=to_t,
                         clower="blue",  # Color for lower da estimation.
                         cupper="red"    # Color for upper da estimation.
                        )
ax[1].set_title('DA vs Turns')
ax[1].legend(prop={'size': 15})

fig.savefig("DA.pdf",bbox_inches='tight')