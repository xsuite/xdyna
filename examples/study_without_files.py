import xdyna as xd
from pathlib import Path



# ================================================================
# === Instantiating DA object and preparing initial conditions ===
# ================================================================

# This will instantiate a DA object (without files)
DA = xd.DA(name='hllhc_da', emittance=2.5e-6, max_turns=1, min_turns=0)

# Generate the initial conditions on a polar grid
# DA.generate_random_initial(num_part=100, r_max=25, px_norm=0, py_norm=0, zeta=0, delta=0.00027)
DA.generate_initial_radial(angles=1, r_min=2, r_max=20, r_step=2/30., delta=0.00027)
# DA.generate_initial_radial(angles=11, r_min=2, r_max=20, r_step=2/30., delta=0.00027, nseeds=60)



# ================================================================
# =================== Getting the xtrack line ====================
# ================================================================

# # Build xtrack line from MAD-X
# # First we specify the path to the line file, so we can choose where it will be stored
# # once calculated (if this step is skipped, it will be asigned the same name as the
# # MAD-X file but as *.line.json)
# DA.line_file = Path.cwd() / 'machines' / 'hllhc_sequence.line.json'
# DA.build_line_from_madx(file=Path.cwd() / 'machines' / 'hl14_col_chrom_15_oct_300_B1_s1.mask', sequence='lhcb1')

# Load an xtrack line (this will also store the path to the line file in the metadata)
DA.load_line_from_file(Path.cwd() / 'machines' / 'hllhc_sequence.line.json')

# # If a line already exists in memory, one can just add it manually:
# DA.line = line



# ================================================================
# ========================== Tracking ============================
# ================================================================

# Do the tracking (over all particles and potentially all seeds)
DA.track_job()

# # In case we are using GPUs, we can manually define a context and build the tracker
# # context = xo.ContextCupy(device=2)
# DA.line.build_tracker(_context=context)
# DA.track_job()



# ================================================================
# ======================= Post-processing ========================
# ================================================================
