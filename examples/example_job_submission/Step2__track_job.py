import sys
import xdyna as xd

# This is an example for a DA study with 60 seeds to be ran on a cluster.
# =======================================================================
# The second step is the tracking, which are the jobs running in parallel.

# We pass the studyname as an argument to the script
study = str(sys.argv[1])
path  = str(sys.argv[2])

# This will load the existing DA based on the meta file with the same name found in the path variable.
# If no path variable is specified, the meta file is assumed to be in the working directory (if this
# is not the case, the script will create a new DA object which is not the intention in this step).
# Because we specified the file containing the xtrack line in Step 1, or calculated it in Step 1bis,
# it will be automatically loaded as well.
DA = xd.DA(name=study, use_files=True, path=path)

# Do the tracking, here for 100 particles.
# The code will automatically look for particles that are not-submitted yet and use these.
DA.track_job(npart=100)

# # In case we are using GPUs, we can manually define a context and build the tracker:
# # context = xo.ContextCupy(device=2)
# DA.line.build_tracker(_context=context)
# DA.track_job(npart=10000)

