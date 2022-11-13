import sys
import xdyna as xd

# This is an example for a DA study with 60 seeds to be ran on a cluster.
# =======================================================================
# The second step is the tracking, which are the jobs running in parallel.

# We pass the studyname as an argument to the script
study = str(sys.argv[1])

# This will load the existing DA based on the meta file with the same name found in the working directory.
# If the script is ran somewhere else, the path to the metafile can be passed with 'path=...'.
# Because we specified the file containing the xtrack line in Step 1, it will be automatically loaded as well.
DA = xd.DA(name=study, use_files=True)

# Do the tracking, here for 100 particles.
# The code will automatically look for particles that are not-submitted yet and use these.
DA.track_job(npart=100)

# In case we are using GPUs, we can manually define a context and build the tracker:
# context = xo.ContextCupy(device=2)
# DA.line.build_tracker(_context=context)
# DA.track_job(npart=10000)

