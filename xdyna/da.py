from scipy import interpolate, integrate
from scipy.constants import c as clight
from math import floor
import numpy as np
import pandas as pd

from pathlib import Path
import json
import os, subprocess
import datetime


import xobjects as xo
import xtrack as xt
import xpart as xp
import sixtracktools as st
    
from .protectfile import ProtectFile
from .da_meta import _DAMetaData


# def load_SixTrack_colldb(filename, *, emit):
#     return CollDB(emit=emit, sixtrack_file=filename)


class DA:
    # The real coordinates have to be generated on the machine that will track them.
    # the absolute truth is the sigma; if this translates to slightly different coordinates
    # then that means those machines also have a slightly different closed orbit etc,
    # so the normalised coordinates are still more correct
#     _surv_cols=[
#                 'x_norm_in',  'px_norm_in',  'y_norm_in',  'py_norm_in',  'zeta_in',  'delta_in',
#                 'x_in',       'px_in',   'y_in',   'py_in',
#                 'ang_xy', 'r_xy',
#                 'ang_xpx', 'ang_ypy', 'r_xpxypy',
#                 'xn_out', 'pxn_out', 'yn_out', 'pyn_out', 'zeta_out', 'delta_out',
#                 'nturns', 'paired_to', 'submitted'
#              ]

    def __init__(self, filename, *, turns=None, nseeds=None, emittance=None, energy=None):
        # Initialise metadata
        self._meta = _DAMetaData(filename=filename)
        if turns is not None:
            self.meta.turns = turns
        if nseeds is not None:
            self.meta.nseeds = nseeds
        if emittance is not None:
            self.meta.emittance = emittance
        if energy is not None:
            self.meta.energy = energy

        self._surv = None
        self._da = None
        self._da_evol = None
        self._active_job = -1
        self._active_job_log = {}


    # =================================================================
    # ================ Generation of intial conditions ================
    # =================================================================

    # Not allowed on parallel process
    def generate_initial_radial(self, *, angles, r_min, r_max, r_step=None, r_num=None, ang_min=None, ang_max=None,
                                px_norm=0, py_norm=0, zeta=0, delta=0.00027,
                                emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        """Generate the initial conditions in a 2D polar grid.
        
        traditionally this is .000000001
        """
        # Does survival already exist?
        if self._surv is not None:
            print("Warning: Initial conditions already exist! No generation done.")
            return
        if self.meta.surv_file.exists():
            raise Exception("Survival parquet file already exists! Cannot generate new initial conditions.")

        # If non-default values are specified, copy them to the metadata
        # In the grid generation further below, only the metadat values
        # should be used (and not the function ones)!
        if emittance is not None:
            self.emittance = emittance
        if self.emittance is None:
            raise ValueError("No emittance defined! Do this first before generating initial conditions")
        if nseeds is not None:
            self.meta.nseeds = nseeds
        if pairs_shift != 0:
            self.meta.pairs_shift = pairs_shift
            if pairs_shift_var is None:
                raise ValueError("Need to set coordinate for the shift between pairs with pairs_shift_var!")
            else:
                self.meta.pairs_shift_var = pairs_shift_var
        elif pairs_shift_var is not None:
            raise ValueError("Need to set magnitude of shift between pairs with pairs_shift!")

        # Make the grid in r
        if r_step is None and r_num is None:
            raise ValueError(f"Specify at least 'r_step' or 'r_num'.")
        elif r_step is not None and r_num is not None:
            raise ValueError(f"Use only one of 'r_step' and 'r_num', not both.")
        elif r_step is not None:
            r_num = floor( (r_max-r_min) / r_step ) + 1
            r_max = r_min + (r_num-1) * r_step
        r = np.linspace(r_min, r_max, r_num )
        # Make the grid in angles
        if ang_min is None and ang_max is None:
            ang_step = 90. / (angles+1)
            ang = np.linspace(ang_step, 90-ang_step, angles )
        else:
            ang_min = 0 if ang_min is None else ang_min
            ang_max = 90 if ang_max is None else ang_max
            ang_step = (ang_max-ang_min) / (angles+1)
            ang = np.linspace(ang_min, ang_max, angles )
        # Make all combinations
        if self.meta.nseeds is None:
            r, ang = np.array(np.meshgrid(r, ang)).reshape(2,-1)
        else:
            self.meta.nseeds = nseeds
            seeds = np.arange(1,nseeds+1)
            ang, seeds, r = np.array(np.meshgrid(ang, seeds, r)).reshape(3,-1)
        # Get the normalised coordinates
        x = r*np.cos(ang*np.pi/180)
        y = r*np.sin(ang*np.pi/180)
        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds is not None:
            self._surv['seed'] = seeds.astype(int)
        self._surv['ang_xy'] = ang
        self._surv['r_xy'] = r
        self._surv['nturns'] = -1
        self._surv['x_norm_in'] = x
        self._surv['y_norm_in'] = y
        self._surv['px_norm_in'] = px_norm
        self._surv['py_norm_in'] = py_norm
        self._surv['zeta_in'] = zeta
        self._surv['delta_in'] = delta
        self._surv['submitted'] = False
        # Create pairs if requested
        if self.meta.pairs_shift != 0:
            coord = self.meta.pairs_shift_var
            # Rename the coordinate correctly
            if coord in ['x', 'y', 'px', 'py']:
                coord += '_norm_in'
                recalc = 'rang'
            elif coord in ['zeta', 'delta']:
                coord += '_in'
            elif coord in ['angle', 'r']:
                coord = coord[:3] + '_xy'
                recalc = 'xy'
            else:
                raise ValueError(f"Value '{self.meta.pairs_shift_var}' not allowed for pairs_shift_var!")
            # The paired particles will be individual rows, with a pointer
            # to the index value of the particle they originate from.
            self._surv['paired_to'] = self._surv.index
            df = self._surv.copy()
            # Add the shift, and recalculate x and y if needed.
            df[coord] += self.meta.pairs_shift
            if recalc == 'xy':
                df['x_norm_in'] = df['r_xy']*np.cos(df['ang_xy']*np.pi/180)
                df['y_norm_in'] = df['r_xy']*np.sin(df['ang_xy']*np.pi/180)
                # Skip recalculation of r and ang, to not screw up analysis of DA later
            # Join existing and new dataframe
            df.set_index(df.index + len(df.index), inplace=True)
            self._surv = pd.concat([self._surv, df])
        with ProtectFile(self.meta.surv_file, 'x+b') as pf:
            self._surv.to_parquet(pf, index=True)
        self.meta.da_type = 'radial'
        self.meta.da_dim = 2



    # =================================================================
    # ==================== Xtrack tracking jobs =======================
    # =================================================================
    
    # Allowed on parallel process
    def xtrack_job(self, npart, turns=None, tracker=None):
        if tracker is not None:
            self._tracker = tracker

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
                line['acsca.d5l4.b1'].voltage = 16e6
                line['acsca.d5l4.b1'].frequency = 400e6
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
            self._warn_job(f"Did not find fort.2 (expected {fort8.as_posix()}).")
        if not fort16.exists():
            self._warn_job(f"Did not find fort.16 (expected {fort16.as_posix()}).")
        fort10  = Path(jobs, f'fort.10_job{job_id}').resolve()
        # Create SixTrack input
        initial = Path(jobs, f'initial_job{job_id}.dat').resolve()
        self._create_sixtrack_initial(initial, seed, use_xtrack=use_xtrack_for_initial)
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


    

    # Note: SixTrack initial.dat is with respect to the closed orbit when using TRAC,
    #       but in the lab frame when using SIMU
    def _create_sixtrack_initial(self, filename, seed, use_xtrack=True):
        if use_xtrack:
            if filename.exists():
                print("Initial conditions already exist for this job.")
                return
            print("Generating initial conditions for this job.")
            # Filename for line
            if seed == '':
                linefile = self.meta.line_file
            else:
                linefile = Path(self.meta.line_file.as_posix().replace('*', seed))
            # Check that line exists if needed:
            if not linefile.exists():
                raise Exception("Using xtrack for generation of initial conditions, but line not yet created!\n" \
                            + "Do this first (on a non-parallel job).")
            with ProtectFile(linefile, 'r') as pf:
                line = xt.Line.from_dict(json.load(pf))
            with ProtectFile(filename, 'x') as pf:
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
                part_xp = np.array(part.px*part.rpp)
                part_yp = np.array(part.py*part.rpp)
                charge = [ round(q) for q in part.charge_ratio*part.q0 ]
                mass_ratio = [ round(m) for m in part.charge_ratio/part.chi ]
                mass = np.array(part.mass0*part.charge_ratio/part.chi)
                part_p = np.array((1+part.delta)*part.p0c)
                # sigma = - beta0*c*dt
                part_dt = - np.array(part.zeta/part.rvv/part.beta0) / clight
                data = pd.DataFrame({
                    'particle ID':   list(range(1, len(self._surv.index)+1)),
                    'parent ID':     list(range(1, len(self._surv.index)+1)),
                    'weight':        1,      # unused
                    'x':             np.array(part.x),
                    'y':             np.array(part.y),
                    'z':             1,      # unused
                    'xp':            part_xp,
                    'yp':            part_yp,
                    'zp':            1,      # unused
                    'mass number':   mass_ratio,        # This is not correct! If the parent particle
                                                        # is an ion, mass_ratio will not approximate
                                                        # the mass number. Works for protons though
                    'atomic number': charge,
                    'mass [GeV/c2]': mass*1e-9,
                    'p [GeV/c2]':    part_p*1e-9,
                    'delta t':       part_dt
                })
                data.to_csv(pf, sep=' ', header=False, index=False)
        else:
            # TODO: give normalised coordinates to SixTrack
            raise NotImplementedError



    # =================================================================
    # ==================== Manage tracking jobs =======================
    # =================================================================

    # Allowed on parallel process
    def _create_job(self, npart, turns):
        if turns is not None:
            if turns > self.meta.turns:
                if self.meta.turns != _DAMetaData._turns_default:
                    print("Warning: This job tracks more turns than foreseen!")
                self.meta.turns = turns
        if self.meta.turns == _DAMetaData._turns_default:
            raise ValueError("Number of tracking turns not set! Cannot track.")
        # Get job ID
        self._active_job = self.meta.new_submission_id()
        with ProtectFile(self.meta.surv_file, 'r+b', wait=0.02) as pf:
            # Get the first npart particle IDs that are not yet submitted
            # TODO: this can probably be optimised by only reading in first
            #       npart lines; no need to read in full file?
            self._surv = pd.read_parquet(pf)
            mask = self._surv['submitted'] == False
            this_part_ids = self._surv[mask].index[:npart]
            # If there are seeds, only take jobs from one seed
            if self.meta.nseeds > 0:
                df = self._surv.loc[this_part_ids]
                seeds = np.unique(df['seed'])
                seed = seeds[0]
                mask = df['seed'] == seed
                this_part_ids = df.loc[mask].index
            else:
                seed = 0
            # Flag them as submitted, before releasing the file again
            self._surv.loc[this_part_ids, 'submitted'] = True
            pf.truncate(0)  # Delete file contents (to avoid appending)
            pf.seek(0)      # Move file pointer to start of file
            self._surv.to_parquet(pf, index=True)
        # Reduce dataframe to only those particles in this job
        self._surv = self._surv.loc[this_part_ids]
        # Submission info
        self._active_job_log = {
                'submission_time': datetime.datetime.now().isoformat(),
                'finished_time':   0,
                'status':          'Running',
                'tracking_turns':  self.meta.turns,
                'particle_ids':    '[' + ', '.join([str(pid) for pid in this_part_ids]) + ']',
                'seed':            int(seed),
                'warnings':        [],
                'output':          ''
        }


    # Allowed on parallel process
    def _update_job_log(self, update):
        self._active_job_log.update(update)
        self.meta.update_submissions(self._active_job, self._active_job_log)

    # Allowed on parallel process
    def _fail_job(self, failtext):
        self._active_job_log.update({'status': 'Failed: ' + failtext})
        self.meta.update_submissions(self._active_job, self._active_job_log)
        raise Exception(failtext)

    # Allowed on parallel process
    def _warn_job(self, warntext):
        self._active_job_log.update({'warnings': self._active_job_log['warnings'] + warntext})
        self.meta.update_submissions(self._active_job, self._active_job_log)
        print(warntext)



    # =================================================================
    # ========================= Calculate DA ==========================
    # =================================================================


    # Not allowed on parallel process
    def calculate_da(self):
        if da_type == 'radial':
            pass
        elif da_type == 'grid':
            pass
        elif da_type in ['radial', 'free']:
            pass # ML

    # =================================================================
    # ============================ Plot DA ============================
    # =================================================================

    # =================================================================
    # ======================= Class attributes ========================
    # =================================================================

    # Not allowed on parallel process
    @property
    def surv_data(self):
        if self._surv is None:
            if self.meta.surv_file.exists():
                with ProtectFile(self.meta.surv_file, 'rb') as pf:
                    self._surv = pd.read_parquet(pf)
            else:
                return None
        if self.da_type == 'radial':
            if self.da_dimension == 2:
                view_cols = ['ang_xy', 'r_xy']
            elif self.da_dimension == 3:
                view_cols = ['ang_xy', 'r_xy', 'delta_in']
            elif self.da_dimension == 4:
                view_cols = ['ang_xy', 'ang_xpx', 'ang_ypy', 'r_xpxypy']
            elif self.da_dimension == 5:
                view_cols = ['ang_xy', 'ang_xpx', 'ang_ypy', 'r_xpxypy', 'delta_in']
            else:
                view_cols = 'all'
        else:
            if self.da_dimension == 2:
                view_cols = ['x_norm_in', 'y_norm_in']
            elif self.da_dimension == 3:
                view_cols = ['x_norm_in', 'y_norm_in', 'delta_in']
            elif self.da_dimension == 4:
                view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in']
            elif self.da_dimension == 5:
                view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in', 'delta_in']
            elif self.da_dimension == 6:
                view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in', 'zeta_in', 'delta_in']
            else:
                view_cols = 'all'
        if view_cols == 'all':
            df = self._surv
        else:
            if self.meta.nseeds > 0:
                view_cols += ['seed']
            if self.meta.pairs_shift == 0:
                view_cols += ['nturns']
                df = self._surv[view_cols]
            else:
                orig = self._surv['paired_to'] == self._surv.index
                df = self._surv.loc[orig,view_cols]
                # Change to np.array to ignore index
                df['nturns1'] = np.array(self._surv.loc[orig,'nturns'])
                df['nturns2'] = np.array(self._surv.loc[~orig,'nturns'])
        return df.rename(columns = {
                    'x_norm_in':'x', 'px_norm_in':'px', 'y_norm_in':'y', 'py_norm_in':'py', 'delta_in':'delta', \
                    'ang_xy':'angle', 'ang_xpx':'angle_x', 'ang_ypy':'angle_y', 'r_xy':'amplitude', 'r_xpxypy': 'amplitude' \
                }, inplace=False)

    @property
    def meta(self):
        return self._meta

    @property
    def da_type(self):
        return self.meta.da_type

    @property
    def da_dimension(self):
        return self.meta._da_dim

    @property
    def emittance(self):
        if self.meta.emitx is None or self.meta.emity is None:
            return None
        else:
            return [self.meta.emitx, self.meta.emity]

    # Not allowed on parallel process
    @emittance.setter
    def emittance(self, emit, update_surv=True):
        oldemittance = self.emittance
        if hasattr(emit, '__iter__'):
            if isinstance(emit, str):
                raise ValueError(f"The emittance has to be a number!")
            elif len(emit) == 2:
                self.meta.emitx = emit[0]
                self.meta.emity = emit[1]
            elif len(emit) == 1:
                self.meta.emitx = emit[0]
                self.meta.emity = emit[0]
            else:
                raise ValueError(f"The emittance must have one or two values (for emitx and emity)!")
        else:
            self.meta.emitx = emit
            self.meta.emity = emit
        # Recalculate initial conditions if set
        if self._surv is not None and oldemittance is not None \
        and self.emittance != oldemittance and update_surv:
            print("Updating emittance")
            corr_x = np.sqrt( oldemittance[0] / self.meta.emitx )
            corr_y = np.sqrt( oldemittance[1] / self.meta.emity )
            with ProtectFile(self.meta.surv_file, 'r+b') as pf:
                self._surv = pd.read_parquet(pf)
                self._surv['x_norm_in']  *= corr_x
                self._surv['px_norm_in'] *= corr_x
                self._surv['y_norm_in']  *= corr_y
                self._surv['py_norm_in'] *= corr_y
                self._surv['r_xy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['y_norm_in']**2 )
                self._surv['r_xpxypy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['px_norm_in']**2 \
                                                + self._surv['y_norm_in']**2 + self._surv['py_norm_in']**2 )
                pf.truncate(0)  # Delete file contents (to avoid appending)
                pf.seek(0)      # Move file pointer to start of file
                self._surv.to_parquet(pf, index=True)

    @property
    def energy(self):
        return self.meta.energy
#     def create_sixtrack_input
    
#     def create_xtrack_input
 
    def convert_to_radial(self):
        # impossible; only to add dimensions or something like that
        raise NotImplementedError

        
        
        

# Function to cut out islands: only keep the turn number if smaller than the previous one.
# Otherwise, replace by previous turn number.
descend = np.frompyfunc(lambda x, y: y if y < x else x, 2, 1)

def _calculate_radial_evo(data):
    # We select the window in number of turns for which each angle has values
    minturns = np.array([ x[1].min() for x in data.values()]).max()
    maxturns = np.array([ x[1].max() for x in data.values()]).min()
    turns = np.unique([ x for x in np.concatenate([ x[1] for x in data.values()]) if (x >= minturns and x <= maxturns) ])
    angles=np.array(list(data.keys()))
    # initialise numpy array
    result = {turn: np.zeros(len(angles)) for turn in turns }
    for i_ang,ang in enumerate(angles):
        rho_surv = np.flip(data[ang])
        # Shift the value of Nturn slightly up such that, in case of a vertical cliff,
        # the interpolation will assign the largest rho_surv value. This cannot be
        # done for the smallest Nturn as this will give an out-of-bounds error.
        rho_surv[0,1:] += 1e-7
        rho_turns = interpolate.interp1d(rho_surv[0], rho_surv[1], kind='linear')(turns)
        for i_turn, turn in enumerate(turns):
            result[turn][i_ang] = rho_turns[i_turn]
    da = np.zeros(len(turns))
    for i_turn, turn in enumerate(turns):
        da[i_turn] = np.sqrt(
            2/np.pi*integrate.trapezoid(x=angles*np.pi/180, y=result[turn]**2)
        )
    return np.array([turns, da])
    
def _get_raw_da_radial(data):
    angles = np.unique(data.angle)
    ampcol = 'amplitude'
    da = {}
    for angle in angles:
        datasort = data[data.angle==angle].sort_values(ampcol)
        # Cut out islands from evolution (i.e. force non-monotonously descending)
        datasort['turns_no_islands'] = descend.accumulate(datasort.turns.astype(object))
        # Keep the values around the places where the number of turns changes
        mask = (datasort.turns_no_islands.diff() < 0 ) | (datasort.turns_no_islands.shift(-1).diff() < 0)
        mask.iloc[-1] = True  # Add the last amplitude step (needed for interpolation later)
        da[angle] = np.array((datasort[mask][ampcol].values,datasort[mask].turns_no_islands.values), dtype=float)
    return da

def get_da_radial(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return np.array([ [ang, vals[0,0]] for ang, vals in _get_raw_da_radial(data).items() ])

def get_da_evo_radial(files):
    data = pd.concat([ pd.read_csv(file) for file in files ])
    return _calculate_radial_evo(_get_raw_da_radial(data))
    


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




