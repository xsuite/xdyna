import xdyna as xd
from pathlib import Path

# This is an example for a DA study with 60 seeds to be ran on a cluster.
# =======================================================================
# In case the line(s) still need(s) to be imported from MAD-X, an extra
# step is needed. As this can potentially take a bit of time, especially
# with 60 seeds, it is maybe beneficial to run this on the cluster as well.

# We pass the studyname as an argument to the script
study = str(sys.argv[1])

# This will load the existing DA based on the meta file with the same name found in the working directory.
# If the script is ran somewhere else, the path to the metafile can be passed with 'path=...'
DA = xd.DA(name=study, use_files=True)

# Now we build the line
DA.build_line_from_madx(sequence='lhcb1')
