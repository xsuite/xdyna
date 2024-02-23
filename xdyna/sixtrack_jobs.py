import os
import subprocess
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import xtrack as xt
import xpart as xp
import xobjects as xo
import sixtracktools as st

from da_meta import _DAMetaData
from da import descend, _calculate_radial_evo
from xaux import ProtectFile


def _get_raw_da_sixdesk(data):
    angles = np.unique(data.angle)
    seeds = np.unique(data.seed)
    ampcol = 'amp'
    da = {}
    data['turns'] = data[['sturns1']] #,'sturns2']].min(axis=1)
    for seed in seeds:
        da[seed] = {}
        for angle in angles:
            datasort = data[(data.angle==angle) & (data.seed==seed)].sort_values(ampcol)
            # Cut out islands from evolution (i.e. force non-monotonously descending)
            datasort['turns_no_islands'] = descend.accumulate(datasort.turns.astype(object))
            # Keep the values around the places where the number of turns changes
            mask = (datasort.turns_no_islands.diff() < 0 ) | (datasort.turns_no_islands.shift(-1).diff() < 0)
            mask.iloc[-1] = True  # Add the last amplitude step (needed for interpolation later)
            da[seed][angle] = np.array((
                                datasort[mask][ampcol].values,datasort[mask].turns_no_islands.values
                            ), dtype=float)
    return da

def get_da_sixdesk(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return {
        seed: np.array([ [ang, vals[0,0]] for ang, vals in da.items() ])
        for seed, da in _get_raw_da_sixdesk(data).items()
    }

def get_da_evo_sixdesk(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return { seed: _calculate_radial_evo(da) for seed,da in _get_raw_da_sixdesk(data).items() }





# =================================================================
# =================== Sixtrack tracking jobs ======================
# =================================================================

# Not allowed on parallel process
# TODO: allow parallel for dedicated MAD-X process
def create_line_from_sixtrack(self, sixtrack_input_folder=None):
    """Create a line from sixtrack input, to be used in parallel tracking"""
    # Checks
    # TODO: check for other variables, such as particle type and RF
    if self.meta.energy == _DAMetaData._energy_default:
        raise ValueError("Energy not set! Do this before creating the line.")
    # Set the SixTrack input folder if not done yet
    self._set_sixtrack_folder(sixtrack_input_folder)
    # List of seeds
    if self.meta.nseeds > 0:
        seeds = list(range(1, self.meta.nseeds+1))
        self.meta.line_file = Path(self.meta.six_path, f'{self.meta.name}.line_*.json')
    else:
        seeds = [''] # Small hack to avoid having to rewrite following loop
        self.meta.line_file = Path(self.meta.six_path, f'{self.meta.name}.line.json')
    # Loop over seeds
    for seed in seeds:
        seed = str(seed)
        seedpath = Path(self.meta.six_path, seed).resolve()
        # If individual fort.3 are missing, make links
        if not Path(seedpath, 'fort.3').exists():
            if Path(self.meta.six_path, 'fort.3').exists():
                Path(seedpath, 'fort.3').symlink_to(Path(self.meta.six_path, 'fort.3'))
            else:
                raise Exception("No fort.3 found!")
        # Filename for line
        if seed == '':
            linefile = self.meta.line_file
        else:
            linefile = Path(self.meta.line_file.as_posix().replace('*', seed))
        if linefile.exists():
            continue
        # Generate line
        with ProtectFile(linefile, 'x') as pf:
            print(f"Calculating line{'' if seed=='' else ' for seed ' + seed}.")
            line = xt.Line.from_sixinput(st.SixInput(seedpath))
            # TODO: no hardcoding of particle selection as proton
            line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=self.meta.energy)
            # TODO: no hardcoding of RF
#                 line['acsca.d5l4.b1'].voltage = 16e6
#                 line['acsca.d5l4.b1'].frequency = 400e6
            json.dump(line.to_dict(), pf, cls=xo.JEncoder)

def _set_sixtrack_folder(self, sixtrack_input_folder=None):
    # Check that folder with SixTrack input files exists
    if sixtrack_input_folder is None:
        if self.meta.six_path is None:
            self.meta.six_path = Path('SixTrack_Files').resolve()
    else:
        self.meta.six_path = Path(sixtrack_input_folder).resolve()
    if not self.meta.six_path.exists():
        raise ValueError(f"SixTrack input folder not found (expected {self.meta.six_path.as_posix()})! Cannot track.")


# Allowed on parallel process
def sixtrack_job(self, *, npart, turns=None, sixtrack_executable=None, sixtrack_input_folder=None,\
                 use_xtrack_for_initial=True):
    """Run a sixtrack job of 'npart' particles for 'nturns'"""
    self._set_sixtrack_folder(sixtrack_input_folder)
    # Check that executable exists and can run
    if sixtrack_executable is None:
        sixtrack_executable = Path(self.meta.six_path,'sixtrack').resolve()
    else:
        sixtrack_executable = Path(sixtrack_executable).resolve()
    if not sixtrack_executable.exists():
        raise ValueError(f"SixTrack executable not found (expected {sixtrack_executable.as_posix()})! Cannot track.")
    if not os.access(sixtrack_executable, os.X_OK):
        raise ValueError("SixTrack executable is not executable! Cannot track.")

    # Create a job: get job ID and start logging
    self._create_job(npart, turns)
    job_id = str(self._active_job)
    self._update_job_log({
            'engine':     'sixtrack',
            'executable': sixtrack_executable.as_posix(),
            'input':      self.meta.six_path.as_posix()
    })

    # Define job script file and output file
    jobs     = Path(self.meta.six_path, f'jobs_{self.meta.name}').resolve()
    jobs.mkdir(exist_ok=True)
    job      = Path(jobs, f'job_{job_id}.sh').resolve()
    job_out  = Path(jobs, f'job_{job_id}.out')
    # Define fort files
    # TODO: a lot is still hardcoded in the fort.3:
    #       npart, turns, pmass, energy, ...
    fort3    = Path(self.meta.six_path, 'fort.3' ).resolve()
    if not fort3.exists():
        self._fail_job(f"Did not find fort.3 (expected {fort3.as_posix()})! Cannot track.")
    seed     = '' if self._active_job_log['seed']==0 else str(self._active_job_log['seed'])
    seedpath = Path(self.meta.six_path, seed).resolve()
    fort2    = Path(seedpath, f'fort.2').resolve()
    fort8    = Path(seedpath, f'fort.8').resolve()
    fort16   = Path(seedpath, f'fort.16').resolve()
    if not fort2.exists():
        self._fail_job(f"Did not find fort.2 (expected {fort2.as_posix()})! Cannot track.")
    if not fort8.exists():
        self._warn_job(f"Did not find fort.8 (expected {fort8.as_posix()}).")
    if not fort16.exists():
        self._warn_job(f"Did not find fort.16 (expected {fort16.as_posix()}).")
    fort10  = Path(jobs, f'fort.10_job{job_id}').resolve()

    # Create SixTrack input
    initial = Path(jobs, f'initial_job{job_id}.dat').resolve()
    if initial.exists():
        print("Initial conditions already exist for this job.")
    else:
        print("Generating initial conditions for this job.")
        linefile = self.meta.line_file if seed=='' else Path(self.meta.line_file.as_posix().replace('*', seed))
        if not linefile.exists():
            raise Exception("Using xtrack for generation of initial conditions, but line not yet created!\n" \
                        + "Do this first (on a non-parallel job).")
        with ProtectFile(linefile, 'r') as pf:
            line = xt.Line.from_dict(json.load(pf))
        tracker = line.build_tracker()
        # TODO: start at different s / element
        part = xp.build_particles(tracker=tracker,
                         # TODO: build_particles does not work with pandas df
                         #       because the latter has a get() attribute (I think)
                         x_norm  = np.array(self._surv['x_norm_in']),
                         px_norm = np.array(self._surv['px_norm_in']),
                         y_norm  = np.array(self._surv['y_norm_in']),
                         py_norm = np.array(self._surv['py_norm_in']),
                         zeta    = np.array(self._surv['zeta_in']),
                         delta   = np.array(self._surv['delta_in']),
                         scale_with_transverse_norm_emitt=(self.meta.emitx, self.meta.emitx)
                     )
#             particles_to_sixtrack_initial(part, initial)

    # Create bash script
    script = '#!/bin/bash\n'
    script += 'temp=$( mktemp -d )\n'
    script += 'cd ${temp}\n'
    script += f'ln -fns {sixtrack_executable.as_posix()} sixtrack\n'
    script += f'ln -fns {fort2.as_posix()} fort.2\n'
    script += f'ln -fns {fort3.as_posix()} fort.3\n'
    script += f'ln -fns {fort8.as_posix()} fort.8\n'
    script += f'ln -fns {fort16.as_posix()} fort.16\n'
    script += f'ln -fns {initial.as_posix()} initial.dat\n'
    script += f'echo "Job {job_id}, seed {seed}" >> {job_out.as_posix()}\n'
    script += f'./sixtrack >> {job_out.as_posix()} 2>&1\n'
    script += f'mv fort.10 {fort10.as_posix()}\n'
    script += 'rm -r ${temp}\n'
    jobf = open(job, "w")
    jobf.write(script)
    jobf.close()
    os.chmod(job, 0o777)
    print("Running sixtrack")
    result = subprocess.run(job, capture_output=True, text=True)
    if result.returncode != 0:
        self._update_job_log({'finished_time':  datetime.datetime.now().isoformat()})
        self._fail_job(self, result.stderr)
    else:
        self._update_job_log({
            'finished_time': datetime.datetime.now().isoformat(),
            'status':        'Finished',
            'output':        result.stdout
        })
        #DO FORT.10


