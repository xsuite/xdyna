import xdyna as xd
from pathlib import Path

# This will instantiate a DA object (without files)
DA = xd.DA(name='hllhc_da_ml', emittance=2.5e-6, max_turns=1e5, min_turns=20, use_files=False)

# Generate the initial conditions on a polar grid
# DA.generate_random_initial(num_part=100, r_max=25, px_norm=0, py_norm=0, zeta=0, delta=0.00027)
DA.generate_initial_radial(angles=11, r_min=2, r_max=20, r_step=2/30., delta=0.00027)
# DA.generate_initial_radial(angles=11, r_min=2, r_max=20, r_step=2/30., delta=0.00027, nseeds=60)

# Load an xtrack line (this will also store the path to the line file in the metadata)
DA.load_line_from_file(Path.cwd() \ 'sequences' \ 'hllhc_sequence.json')

# If a line already exists in memory, one can just add it manually:
# DA.line = line

# Do the tracking (over all particles and potentially all seeds)
DA.track_job()

# In case we are using GPUs, we can manually define a context and build the tracker
# context = xo.ContextCupy(device=2)
# DA.line.build_tracker(_context=context)
# DA.track_job()
