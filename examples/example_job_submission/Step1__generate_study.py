import sys
from pathlib import Path

import xdyna as xd

# This is an example for a DA study with 60 seeds to be ran on a cluster.
# =======================================================================
# The first step is to generate initial conditions, which cannot be ran
# in parallel. To allow for the tracking jobs to know what to do and then
# to be ran in parallel, it is imperative to set the flag use_files to True.

# We pass the studyname as an argument to the script
study = str(sys.argv[1])

# This will instantiate a DA object, and store the metadata in a *.meta.json file
DA = xd.DA(name=study, normalised_emittance=2.5e-6, max_turns=1e5, min_turns=20, use_files=True)

# Generate the initial conditions on a polar grid.
# This will store the particle coordinates in a common database file.
DA.generate_initial_radial(angles=11, r_min=2, r_max=20, r_step=2/30., delta=0.00027, nseeds=60)

# # If we already have a file containing the xtrack line(s), we can give its path to the DA object,
# # such that it can be loaded by the tracking jobs:
# DA.line_file = Path.cwd() / 'machines' / 'hllhc_sequence_with_seeds.json'

# Otherwise we specify the path to the mad-x file and go to Step1bis. We also specify the path
# to the line file, so we can choose where it will be stored once calculated.
DA.madx_file = Path.cwd() / 'machines' / 'hl14_col_chrom_15_oct_300_B1.mask'
DA.line_file = Path.cwd() / 'machines' / 'hllhc_sequence_with_seeds.line.json'
