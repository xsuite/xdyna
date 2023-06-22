import xdyna as xd
from pathlib import Path



# ================================================================
# === Instantiating DA object and preparing initial conditions ===
# ================================================================

# This will instantiate a DA object (without files)
DA = xd.DA(name='hllhc_da', normalised_emittance=2.5e-6, max_turns=1e4)

# Generate the initial conditions on a polar grid
# DA.generate_random_initial(num_part=100, r_max=25, px_norm=0, py_norm=0, zeta=0, delta=0.00027)
DA.generate_initial_radial(angles=11, r_min=2, r_max=20, r_step=2/30., delta=0.00027)
# DA.generate_initial_radial(angles=11, r_min=2, r_max=20, r_step=2/30., delta=0.00027, nseeds=60)



# ================================================================
# =================== Getting the xtrack line ====================
# ================================================================

# # Build xtrack line from MAD-X
# # First we specify the path to the line file, so we can choose where it will be stored
# # once calculated (if this step is skipped, it will be asigned the same name as the
# # MAD-X file but as *.line.json)
# DA.line_file = Path.cwd() / 'test.line.json'
# DA.build_line_from_madx(file=Path.cwd() / 'machines' / 'hl14_col_chrom_15_oct_300_B1_s1.mask', sequence='lhcb1')

# Load an xtrack line (this will also store the path to the line file in the metadata)
DA.load_line_from_file(xd._pkg_root / '..' / 'examples' / 'machines' / 'hllhc_sequence.line.json')

# # If a line already exists in memory, one can just add it manually:
# DA.line = line



# ================================================================
# ========================== Tracking ============================
# ================================================================

# Do the tracking (over all particles and potentially all seeds)
# Watch out, with the current configuration (~3000 particles) this takes around 1.5s per turn,
# or, in other words, a total of almost 3 hours for 1e4 turns!
DA.track_job()

# # In case we are using GPUs, we can manually define a context and build the tracker
# # context = xo.ContextCupy(device=2)
# DA.line.build_tracker(_context=context)
# DA.track_job()



# ================================================================
# ======================= Post-processing ========================
# ================================================================
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