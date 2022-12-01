import sys
from pathlib import Path
import xdyna as xd

# This is an example for a DA study with 60 seeds to be ran on a cluster.
# =======================================================================
# In case the line(s) still need(s) to be imported from MAD-X, an extra
# step is needed. As this can potentially take a bit of time, especially
# with 60 seeds, it is maybe beneficial to run this on the cluster as well.

# We pass the studyname and path as an argument to the script
study = str(sys.argv[1])
path  = str(sys.argv[2])

# This will load the existing DA based on the meta file with the same name found in the path variable.
# If no path variable is specified, the meta file is assumed to be in the working directory (if this
# is not the case, the script will create a new DA object which is not the intention in this step).
DA = xd.DA(name=study, use_files=True, path=path)

# Now we build the line
DA.build_line_from_madx(sequence='lhcb1')
