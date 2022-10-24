from scipy import interpolate, integrate
# from scipy.constants import c as clight
from math import floor
import numpy as np
from numpy.random import default_rng
import pandas as pd

# from pathlib import Path
# import json
# import os, subprocess
import datetime
import time

import xobjects as xo
# import xtrack as xt
import xpart as xp

from .protectfile import ProtectFile
from .da_meta import _DAMetaData
from .geometry import _bleed, distance_to_polygon_2D



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

    def __init__(self, filename, *, min_turns=None, max_turns=None, nseeds=0, emittance=None, energy=None,
                 load_files=False, memory_threshold=1e9):
        # Initialise metadata
        self._meta = _DAMetaData(filename=filename)
        self.meta.nseeds = nseeds
        if min_turns is not None:
            self.meta.min_turns = min_turns
        if max_turns is not None:
            self.meta.max_turns = max_turns
        if emittance is not None:
            self.emittance = emittance
        if energy is not None:
            self.meta.energy = energy
        self.memory_threshold = memory_threshold

        self._surv = None
        self._da = None
        self._da_evol = None
        self._active_job = -1
        self._active_job_log = {}
        if load_files:
            if self.meta.surv_file.exists():
                with ProtectFile(self.meta.surv_file, 'rb') as pf:
                    self._surv = pd.read_parquet(pf)
            if self.meta.da_file.exists():
                with ProtectFile(self.meta.da_file, 'rb') as pf:
                    self._da = pd.read_parquet(pf)
            if self.meta.da_evol_file.exists():
                with ProtectFile(self.meta.da_evol_file, 'rb') as pf:
                    self._da_evol = pd.read_parquet(pf)


    # =================================================================
    # ================ Generation of intial conditions ================
    # =================================================================

    def _prepare_generation(self, emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
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

    def _create_pairs(self):
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



    # Not allowed on parallel process
    def generate_random_initial(self, *, num_part=1000, r_max=25, px_norm=0, py_norm=0, zeta=0, delta=0.00027,
                                emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        """Generate the initial conditions in a 2D random grid.

        traditionally this is .000000001
        """
        self._prepare_generation(emittance, nseeds, pairs_shift, pairs_shift_var)
        # Make the data
        rng = default_rng()
        if self.meta.nseeds > 0:
            r = rng.uniform(low=0, high=r_max**2, size=num_part*self.meta.nseeds)
            r = np.sqrt(r)
            th = rng.uniform(low=0, high=2*np.pi, size=num_part*self.meta.nseeds)
            x = r*np.cos(th)
            y = r*np.sin(th)
            seeds = np.repeat(np.arange(1,self.meta.nseeds+1), num_part)
        else:
            r = rng.uniform(low=0, high=r_max**2, size=num_part)
            r = np.sqrt(r)
            th = rng.uniform(low=0, high=2*np.pi, size=num_part)
            x = r*np.cos(th)
            y = r*np.sin(th)

        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
            self._surv['seed'] = seeds.astype(int)
        self._surv['x_norm_in'] = x
        self._surv['y_norm_in'] = y
        self._surv['nturns'] = -1
        self._surv['px_norm_in'] = px_norm
        self._surv['py_norm_in'] = py_norm
        self._surv['zeta_in'] = zeta
        self._surv['delta_in'] = delta
        self._surv['x_out'] = -1
        self._surv['y_out'] = -1
        self._surv['px_out'] = -1
        self._surv['py_out'] = -1
        self._surv['zeta_out'] = -1
        self._surv['delta_out'] = -1
        self._surv['s_out'] = -1
        self._surv['state'] = 1
        self._surv['submitted'] = False
        self._surv['finished'] = False
        self._create_pairs()
        with ProtectFile(self.meta.surv_file, 'x+b') as pf:
            self._surv.to_parquet(pf, index=True)
        self.meta.da_type = 'monte_carlo'
        self.meta.da_dim = 2
        self.meta.r_max = r_max



    # Not allowed on parallel process
    def generate_initial_radial(self, *, angles, r_min, r_max, r_step=None, r_num=None, ang_min=None, ang_max=None,
                                px_norm=0, py_norm=0, zeta=0, delta=0.00027,
                                emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        """Generate the initial conditions in a 2D polar grid.

        traditionally this is .000000001
        """
        self._prepare_generation(emittance, nseeds, pairs_shift, pairs_shift_var)

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
        if self.meta.nseeds > 0:
            self.meta.nseeds = nseeds
            seeds = np.arange(1,nseeds+1)
            ang, seeds, r = np.array(np.meshgrid(ang, seeds, r)).reshape(3,-1)
        else:
            r, ang = np.array(np.meshgrid(r, ang)).reshape(2,-1)
        # Get the normalised coordinates
        x = r*np.cos(ang*np.pi/180)
        y = r*np.sin(ang*np.pi/180)
        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
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
        self._surv['x_out'] = -1
        self._surv['y_out'] = -1
        self._surv['px_out'] = -1
        self._surv['py_out'] = -1
        self._surv['zeta_out'] = -1
        self._surv['delta_out'] = -1
        self._surv['s_out'] = -1
        self._surv['state'] = 1
        self._surv['submitted'] = False
        self._surv['finished'] = False
        self._create_pairs()
        with ProtectFile(self.meta.surv_file, 'x+b') as pf:
            self._surv.to_parquet(pf, index=True)
        self.meta.da_type = 'radial'
        self.meta.da_dim = 2
        self.meta.r_max = r_max


    # Not allowed on parallel process
    def add_random_initial(self, *, num_part=5000, min_turns=None):
        from .ml import MLBorder

        # TODO: make compatible with seeds and with pairs
        if self.meta.nseeds > 0 or self.meta.pairs_shift != 0:
            raise NotImplementedError

        # TODO: erase already calculated DA values

        # Set minimum turns
        if min_turns is None:
            if self.meta.min_turns is None:
                self.meta.min_turns = 20
        else:
            if self.meta.min_turns is not None and self.meta.min_turns != turns:
                # TODO: This should only be checked if we already re-sampled.
                # There is no harm by changing min_turns after the initial run
                print(f"Warning: 'min_turns' can be set only once (and is already set to {self.meta.min_turns}). "
                      + "Ignored the new value.")

        # Get existing results
        if self.surv_data is None:
            raise ValueError("No survival data found!")
        data = self.surv_data
        # TODO: check that all px_norm etc are the same
        px_norm = np.unique(self._surv['px_norm_in'])[0]
        py_norm = np.unique(self._surv['py_norm_in'])[0]
        zeta = np.unique(self._surv['zeta_in'])[0]
        delta = np.unique(self._surv['delta_in'])[0]

        # Get minimum and maximum borders
        prev = time.process_time()
        print(f"Getting minimum border (at Nmin={self.meta.min_turns} turns)... ", end='')
        ML_min = MLBorder(data.x, data.y, data.nturns, at_turn=self.meta.min_turns, memory_threshold=self.memory_threshold)
        ML_min.fit(50)
        ML_min.evaluate(0.5)
        min_border_x, min_border_y = ML_min.border
        print(f"done (in {round(time.process_time()-prev,2)} seconds).")

        prev = time.process_time()
        print(f"Getting maximum border (at Nmax={self.meta.max_turns} turns)... ", end='')
        ML_max = MLBorder(data.x, data.y, data.nturns, at_turn=self.meta.max_turns, memory_threshold=self.memory_threshold)
        ML_max.fit(50)
        ML_max.evaluate(0.5)
        max_border_x, max_border_y = ML_max.border
        print(f"done (in {round(time.process_time()-prev,2)} seconds).")

        # TODO: catch when border tells us that not enough samples are available and larger r or larger Nmin is needed

        print(f"Generating samples... ", end='')
        # First we create a 'pool' of samples that contains 100 times as many particles as wished.
        # This is to get a reliable distribution to select from later.
        # We make this pool by sampling a uniform ring, from the inner circle of the max_border (with bleed)
        # until the outer circle of the min_border (with bleed). Then we veto particles if they're not
        # within the borders (with bleed)
        rng = default_rng()
        _, r_max, bleed_min_val = _bleed(min_border_x, min_border_y, margin=0.1)
        r_min, _, bleed_max_val = _bleed(max_border_x, max_border_y, margin=0.1)

        r_min -= bleed_max_val
        r_max += bleed_min_val
        r_max = min(r_max, self.meta.r_max)

        n_pool = num_part * 100
        x_pool = np.array([])
        y_pool = np.array([])
        d_pool = np.array([])

        # Go in chunks of maximum ram during calculation of distances
        n_samples = int(self.memory_threshold / 25 / max(len(min_border_x), len(max_border_x)))

        while n_pool > 0:
            n_samples = min(n_samples, int(2.5*n_pool))
            r = rng.uniform(low=r_min**2, high=r_max**2, size=n_samples)
            r = np.sqrt(r)
            th = rng.uniform(low=0, high=2*np.pi, size=n_samples)
            x = r*np.cos(th)
            y = r*np.sin(th)
            distance_to_min = distance_to_polygon_2D(x, y, min_border_x, min_border_y)
            distance_to_max = distance_to_polygon_2D(x, y, max_border_x, max_border_y)

            # We only sample points in between min_border and max_border, but also allow
            # for a bit of 'bleed' over the border.
            between   = (distance_to_min <= 0) & (distance_to_max > 0)
            bleed_min = (distance_to_min > 0) & (distance_to_min < bleed_min_val)
            bleed_max = (distance_to_max <= 0) & (distance_to_max > -bleed_max_val)

            x_clean = x[bleed_max | bleed_min | between]
            y_clean = y[bleed_max | bleed_min | between]

            # Rescale the distances to adapt to the probabilities later:
            distance = distance_to_max.copy()
            # Those that bleeded over the maximum border are rescaled between 0 (on the border) and -1/2 (maximum bleed)
            distance[bleed_max] /= bleed_max_val*2
            # Those in between are rescaled to the distance between the two borders: 0 (on the max border), 1 (on the min border)
            distance[between] /= distance_to_max[between] - distance_to_min[between]
            # Those that bleeded over the minimum border are rescaled between 1 (on the border) and 3/2 (maximum bleed)
            distance[bleed_min] = distance_to_min[bleed_min]/bleed_min_val/2 + 1

            d_clean = distance[bleed_max | bleed_min | between]

            n_pool -= len(x_clean)
            if n_pool < 0:
                x_pool = np.concatenate([x_pool, x_clean[:n_pool]])
                y_pool = np.concatenate([y_pool, y_clean[:n_pool]])
                d_pool = np.concatenate([d_pool, d_clean[:n_pool]])
            else:
                x_pool = np.concatenate([x_pool, x_clean])
                y_pool = np.concatenate([y_pool, y_clean])
                d_pool = np.concatenate([d_pool, d_clean])

        # The pool is generated, now we sample!
        # As a probability distribution, we use a gaussian scaled by a square root, shifted 1/2 to the left
        prob = np.sqrt(d_pool+0.5)*np.exp(-(d_pool+0.5)**2)
        prob /= prob.sum()
        ids = rng.choice(np.arange(len(x_pool)), size=num_part, replace=False, p=prob, shuffle=False)
        x_new = x_pool[ids]
        y_new = y_pool[ids]
        print(f"done (in {round(time.process_time()-prev,2)} seconds).")

        # Make new dataframe
        df_new = pd.DataFrame()
#         if self.meta.nseeds > 0:
#             self._surv['seed'] = seeds.astype(int)
        df_new['x_norm_in'] = x_new
        df_new['y_norm_in'] = y_new
        df_new['nturns'] = -1
        df_new['px_norm_in'] = px_norm
        df_new['py_norm_in'] = py_norm
        df_new['zeta_in'] = zeta
        df_new['delta_in'] = delta
        df_new['x_out'] = -1
        df_new['y_out'] = -1
        df_new['px_out'] = -1
        df_new['py_out'] = -1
        df_new['zeta_out'] = -1
        df_new['delta_out'] = -1
        df_new['s_out'] = -1
        df_new['state'] = 1
        df_new['submitted'] = False
        df_new['finished'] = False
#         self._create_pairs()   # Needs to only act on non-finished particles ...        
        self._surv = pd.concat([
                            self._surv,
                            df_new
                    ], ignore_index=True)

        with ProtectFile(self.meta.surv_file, 'wb') as pf:
            self._surv.to_parquet(pf, index=True, engine="pyarrow")



    # =================================================================
    # ==================== Xtrack tracking jobs =======================
    # =================================================================

    # TODO: if job has failed, it remains on submitted=True and finished=False so it won't resubmit.
    # How to change this?

    # Allowed on parallel process
    def track_job(self, *,  npart, tracker, turns=None, resubmit_unfinished=False, logging=False):
        if tracker is None:
            raise ValueError()
            
        # Create a job: get job ID and start logging
        part_ids = self._create_job(npart, turns, resubmit_unfinished, logging)
        job_id = str(self._active_job)

        # Create initial particles
        x_norm  = self._surv['x_norm_in'].to_numpy()
        y_norm  = self._surv['y_norm_in'].to_numpy()
        px_norm = self._surv['px_norm_in'].to_numpy()
        py_norm = self._surv['py_norm_in'].to_numpy()
        zeta    = self._surv['zeta_in'].to_numpy()
        delta   = self._surv['delta_in'].to_numpy()

        context=tracker._buffer.context
        part = xp.build_particles(_context=context,
                          tracker=tracker,
                          x_norm=x_norm, y_norm=y_norm, px_norm=px_norm, py_norm=py_norm, zeta=zeta, delta=delta,
                          scale_with_transverse_norm_emitt=self.emittance
                         )
        # Track
        if logging:
            self._append_job_log('output', datetime.datetime.now().isoformat() + '  Start tracking job ' + str(job_id) + '.')
        tracker.track(particles=part, num_turns=self.this_turns)
        context.synchronize()
        if logging:
            self._append_job_log('output', datetime.datetime.now().isoformat() + '  Done tracking job ' + str(job_id) + '.')

        # Store results
        part_id   = context.nparray_from_context_array(part.particle_id)
        sort      = np.argsort(part_id)
        x_out     = context.nparray_from_context_array(part.x)[sort]
        y_out     = context.nparray_from_context_array(part.y)[sort]
        survturns = context.nparray_from_context_array(part.at_turn)[sort]
        px_out    = context.nparray_from_context_array(part.px)[sort]
        py_out    = context.nparray_from_context_array(part.py)[sort]
        zeta_out  = context.nparray_from_context_array(part.zeta)[sort]
        delta_out = context.nparray_from_context_array(part.delta)[sort]
        s_out     = context.nparray_from_context_array(part.s)[sort]
        state     = context.nparray_from_context_array(part.state)[sort]

        with ProtectFile(self.meta.surv_file, 'r+b', wait=0.02) as pf:
            full_surv = pd.read_parquet(pf)
            full_surv.loc[part_ids, 'finished'] = True
            full_surv.loc[part_ids, 'x_out'] = x_out
            full_surv.loc[part_ids, 'y_out'] = y_out
            full_surv.loc[part_ids, 'nturns'] = survturns.astype(np.int64)
            full_surv.loc[part_ids, 'px_out'] = px_out
            full_surv.loc[part_ids, 'py_out'] = py_out
            full_surv.loc[part_ids, 'zeta_out'] = zeta_out
            full_surv.loc[part_ids, 'delta_out'] = delta_out
            full_surv.loc[part_ids, 's_out'] = s_out
            full_surv.loc[part_ids, 'state'] = state
            pf.truncate(0)  # Delete file contents (to avoid appending)
            pf.seek(0)      # Move file pointer to start of file
            full_surv.to_parquet(pf, index=True)
        self._surv = full_surv

        if logging:
            self._update_job_log({
                'finished_time': datetime.datetime.now().isoformat(),
                'status': 'Finished'
            })



    # =================================================================
    # ==================== Manage tracking jobs =======================
    # =================================================================

    # Allowed on parallel process
    def _create_job(self, npart, turns, resubmit_unfinished=False, logging=False):
        if turns is not None:
            if self.meta.max_turns is None:
                self.meta.max_turns = turns
            elif turns < self.meta.max_turns:
                print("Warning: The argument 'turns' was set a value lower than DA.max_turns! "
                      + "Ignored the former.")
            elif turns > self.meta.max_turns:
#                 if self.meta.max_turns != _DAMetaData._turns_default:
#                 self.this_turns = turns
                raise NotImplementedError
        self.this_turns = self.meta.max_turns
        if self.this_turns == _DAMetaData._max_turns_default:
            raise ValueError("Number of tracking turns not set! Cannot track.")
        # Get job ID
        self._active_job = self.meta.new_submission_id() if logging else 0
        with ProtectFile(self.meta.surv_file, 'r+b', wait=0.02) as pf:
            # Get the first npart particle IDs that are not yet submitted
            # TODO: this can probably be optimised by only reading last column
            self._surv = pd.read_parquet(pf)
            # TODO: this doesn't work as multiple jobs will do the same particles..
#             if resubmit_unfinished:
#                 mask = self._surv['finished'] == False
#             else:
#                 mask = self._surv['submitted'] == False
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
            # Quit the job if no particles need to be submitted
            if len(this_part_ids) == 0:
                print("No more particles to submit! Exiting...")
                # TODO: this doesn't work!
                if logging:
                    self.meta.update_submissions(self._active_job, {'status': 'No submission needed.'})
                exit()
            # Otherwise, flag the particles as submitted, before releasing the file again
            self._surv.loc[this_part_ids, 'submitted'] = True
            pf.truncate(0)  # Delete file contents (to avoid appending)
            pf.seek(0)      # Move file pointer to start of file
            self._surv.to_parquet(pf, index=True)
        # Reduce dataframe to only those particles in this job
        self._surv = self._surv.loc[this_part_ids]
        # Submission info
        if logging:
            self._active_job_log = {
                    'submission_time': datetime.datetime.now().isoformat(),
                    'finished_time':   0,
                    'status':          'Running',
                    'tracking_turns':  self.this_turns,
                    'particle_ids':    '[' + ', '.join([str(pid) for pid in this_part_ids]) + ']',
                    'seed':            int(seed),
                    'warnings':        [],
                    'output':          [],
            }
        return this_part_ids


    # Allowed on parallel process
    def _update_job_log(self, update):
        self._active_job_log.update(update)
        self.meta.update_submissions(self._active_job, self._active_job_log)

    # Allowed on parallel process
    def _append_job_log(self, key, update):
        self._active_job_log[key].append(update)
        self.meta.update_submissions(self._active_job, self._active_job_log)

    # Allowed on parallel process
    def _fail_job(self, failtext):
        self._active_job_log['status'] = 'Failed: ' + failtext
        self.meta.update_submissions(self._active_job, self._active_job_log)
        raise Exception(failtext)

    # Allowed on parallel process
    def _warn_job(self, warntext):
        self._active_job_log['warnings'].append(warntext)
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
        elif da_type in ['monte_carlo', 'free']:
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
        if hasattr(self,'_surv') and self._surv is not None and oldemittance is not None \
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
    



