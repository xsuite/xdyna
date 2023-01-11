# xdyna

Tools to study beam dynamics in xtrack simulations, like dynamic aperture calculations, PYTHIA integration, dynamic indicators, ...

## Dynamic aperture studies

The `xdyna` package provides the `DA` class which serves as a simple front-end for setting up and running dynamic aperture studies.

To start, a `xtrack.line` object is required.
The following code then sets up the study and launches the tracking

```python

import xdyna as xd

da = xd.DA(
    name='name_of_your_study', # used to generate a directory where files are stored
    normalised_emittance=[1,1], # provide normalized emittance for particle initialization in [m]
    max_turns=1e5, # number of turns to track
    use_files=False 
    # in case DA studies should run on HTC condor, files are used to collect the information
    # if the tracking is performed locally, no files are needed
)
    
# initialize a grid of particles using 5 angles in x-y space, in a range from 0 to 20 sigmas in steps of 5 sigma.
da.generate_initial_radial(angles=5, r_min=0, r_max=20, r_step=5, delta=0.) 

da.line = line # associate prev. created line, holding the lattice and context, with DA object

da.track_job() # start the tracking

da.survival_data # returns a dataframe with the number of survived turns for the initial position of each particle

```

To use on a platform like HTCondor, perform the same setup as before but using `use_files=True`.
Each HTCondor job then only requires the following lines

```python
import xdyna as xd
# This will load the existing DA based on the meta file with the same name found in the working directory.
# If the script is ran somewhere else, the path to the metafile can be passed with 'path=...'.
DA = xd.DA(name=study, use_files=True)

# Do the tracking, here for 100 particles.
# The code will automatically look for particles that are not-submitted yet and use these.
DA.track_job(npart=100)
```
