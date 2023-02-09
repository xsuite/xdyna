from math import floor
from pathlib import Path
import json
import datetime
import time
import tempfile

from scipy import interpolate, integrate
# from scipy.constants import c as clight
import numpy as np
from numpy.random import default_rng
import pandas as pd

import xobjects as xo
import xtrack as xt
import xpart as xp

from .protectfile import ProtectFile
from .da_meta import _DAMetaData
from .geometry import _bleed, distance_to_polygon_2D


_db_access_wait_time = 0.02

# TODO: Function to generate DA object from meta file
#       def generate_from_file(filename):
#           DAMetaData(name = child, path= parent)    check that file ends in .meta.json etc




class DA:
    # The real coordinates have to be generated on the machine that will track them (ideally at least).
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

    def __init__(self, name, *, path=Path.cwd(), use_files=False, read_only=False, **kwargs):
        # Initialise metadata
        self._meta = _DAMetaData(name=name, path=path, use_files=use_files, read_only=read_only)
        if self.meta._new and not read_only:
            self.meta._store_properties = False
            self.meta.nseeds       = kwargs.pop('nseeds',       _DAMetaData._defaults['nseeds'])
            if 'max_turns' not in kwargs.keys():
                raise TypeError("DA.__init__() missing 1 required positional argument: 'max_turns'")
            self.meta.max_turns    = kwargs.pop('max_turns')
            self.meta.min_turns    = kwargs.pop('min_turns',    _DAMetaData._defaults['min_turns'])
            self.meta.energy       = kwargs.pop('energy',       _DAMetaData._defaults['energy'])
            self.meta.db_extension = kwargs.pop('db_extension', _DAMetaData._defaults['db_extension'])
            nemitt_x               = kwargs.pop('nemitt_x',     None)
            nemitt_y               = kwargs.pop('nemitt_y',      None)
            normalised_emittance   = kwargs.pop('normalised_emittance', None)
            if normalised_emittance is not None:
                if nemitt_x is not None or nemitt_y is not None:
                    raise ValueError("Use either normalised_emittance, or nemitt_x and nemitt_y.")
                self.update_emittance(normalised_emittance, update_surv=False)
            elif nemitt_x is not None or nemitt_y is not None:
                if nemitt_x is not None and nemitt_y is not None:
                    self.update_emittance([nemitt_x, nemitt_y], update_surv=False)
                else:
                    raise ValueError("Need both nemitt_x and nemitt_y.")
            self.meta._store()

        # Initialise DA data
        self.memory_threshold = kwargs.pop('memory_threshold', 1e9)
        self._surv = None
        self._da = None
        self._da_evol = None
        self._active_job = -1
        self._active_job_log = {}
        self._line = None
        self.read_surv()
        self.read_da()
        self.read_da_evol()

        # Ignore leftover arguments
        if kwargs != {}:
            print(f"Ignoring unused arguments {list(kwargs.keys())}!")



    # =================================================================
    # ======================= Class attributes ========================
    # =================================================================

    @property
    def survival_data(self):
        if self._surv is None:
            self.read_surv()
        if self._surv is None:  # also nothing on file
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
            elif self.da_dimension == 6:
                view_cols = ['ang_xy', 'ang_xpx', 'ang_ypy', 'r_xpxypy', 'zeta_in', 'delta_in']
        elif self.da_type == 'grid':
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
            view_cols = ['x_norm_in', 'px_norm_in', 'y_norm_in', 'py_norm_in', 'zeta_in', 'delta_in']
        
        if self.meta.nseeds > 0:
            view_cols += ['seed']
        
        if self.meta.pairs_shift == 0:
            view_cols += ['nturns', 'state']
            df = self._surv[view_cols]
        else:
            orig = self._surv['paired_to'] == self._surv.index
            df = self._surv.loc[orig,view_cols]
            # Change to np.array to ignore index
            df['nturns1'] = np.array(self._surv.loc[orig,'nturns'])
            df['nturns2'] = np.array(self._surv.loc[~orig,'nturns'])
            df['state1'] = np.array(self._surv.loc[orig,'state'])
            df['state2'] = np.array(self._surv.loc[~orig,'state'])
        
        return df.rename(columns = {
                    'x_norm_in':'x', 'px_norm_in':'px', 'y_norm_in':'y', 'py_norm_in':'py', 'delta_in':'delta', \
                    'ang_xy':'angle', 'ang_xpx':'angle_x', 'ang_ypy':'angle_y', 'r_xy':'amplitude', 'r_xpxypy': 'amplitude' \
                }, inplace=False)


    def to_pandas(self, full: bool=False) -> pd.DataFrame:
        if full:
            return self._surv
        return self.survival_data

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
    def max_turns(self):
        return self.meta.max_turns

    @max_turns.setter
    def max_turns(self, max_turns):
        if max_turns <= self.meta.max_turns:
            print("Warning: new value for max_turns smaller than or equal to existing max_turns! Nothing done.")
            return
        # Need to flag particles that survived to previously defined max_turns as to be resubmitted
        # TODO: need to test
        self.reset_survival_output(mask_by='nturns', mask_func=(lambda x: x == self.meta.max_turns))

    @property
    def nemitt_x(self):
        return self.meta.nemitt_x

    @property
    def nemitt_y(self):
        return self.meta.nemitt_y

    @property
    def normalised_emittance(self):
        if self.nemitt_x is None or self.nemitt_y is None:
            return None
        else:
            return [self.nemitt_x, self.nemitt_y]

    # Not allowed on parallel process
    def update_emittance(self, emit, update_surv=True):
        oldemittance = self.normalised_emittance
        if hasattr(emit, '__iter__'):
            if isinstance(emit, str):
                raise ValueError(f"The emittance has to be a number!")
            elif len(emit) == 2:
                self.meta.nemitt_x = emit[0]
                self.meta.nemitt_y = emit[1]
            elif len(emit) == 1:
                self.meta.nemitt_x = emit[0]
                self.meta.nemitt_y = emit[0]
            else:
                raise ValueError(f"The emittance must have one or two values (for nemitt_x and nemitt_y)!")
        else:
            self.meta.nemitt_x = emit
            self.meta.nemitt_y = emit
        # Recalculate initial conditions if set
        if update_surv and self._surv is not None and oldemittance is not None and self.normalised_emittance != oldemittance:
            print("Updating emittance")
            corr_x = np.sqrt( oldemittance[0] / self.nemitt_x )
            corr_y = np.sqrt( oldemittance[1] / self.nemitt_y )
            if self.surv_exists():
                with ProtectFile(self.meta.surv_file, 'r+b') as pf:
                    self.read_surv(pf)
                    self._surv['x_norm_in']  *= corr_x
                    self._surv['px_norm_in'] *= corr_x
                    self._surv['y_norm_in']  *= corr_y
                    self._surv['py_norm_in'] *= corr_y
                    self._surv['r_xy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['y_norm_in']**2 )
                    self._surv['r_xpxypy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['px_norm_in']**2 \
                                                    + self._surv['y_norm_in']**2 + self._surv['py_norm_in']**2 )
                    self.write_surv(pf)
            elif self._surv is not None:
                self._surv['x_norm_in']  *= corr_x
                self._surv['px_norm_in'] *= corr_x
                self._surv['y_norm_in']  *= corr_y
                self._surv['py_norm_in'] *= corr_y
                self._surv['r_xy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['y_norm_in']**2 )
                self._surv['r_xpxypy'] = np.sqrt( self._surv['x_norm_in']**2 + self._surv['px_norm_in']**2 \
                                                + self._surv['y_norm_in']**2 + self._surv['py_norm_in']**2 )

    @property
    def line(self):
        return self._line
            
    @line.setter
    def line(self, line):
        if line is None:
            if self.meta.line_file == -1:
                self.meta.line_file = None  # The line is no longer manually added
            self._line = None
            self.meta.energy = None
        else:
            self.meta.line_file = -1  # Mark line as manually added
            self._line = line
            if self.meta.nseeds == 0:
                # No seeds, so self.line is just the line
                if not isinstance(line, xt.Line):
                    self._line = xt.Line.from_dict(line)
                self.meta.energy = self.line.particle_ref.p0c[0]
            else:
                # Seeds, so line is a dict of lines
                energy = []
                for seed, this_line in line.items():
                    if not isinstance(this_line, xt.Line):
                        self._line[seed] = xt.Line.from_dict(this_line)
                    energy = [ *energy, self._line[seed].particle_ref.p0c[0] ]
                if len(np.unique([ round(e, 4) for e in energy])) != 1:
                    raise ValueError(f"The lines for the different seeds have different energies: {energy}!")
                self.meta.energy = energy[0]

    @property
    def madx_file(self):
        return self.meta.madx_file

    @madx_file.setter
    def madx_file(self, file):
        self.meta.madx_file = file

    @property
    def line_file(self):
        if self.meta.line_file == -1:
            return 'line manually added'
        else:
            return self.meta.line_file

    @line_file.setter
    def line_file(self, file):
        self.meta.line_file = file

    def load_line_from_file(self, file=None):
        if file is not None and self.line_file is not None and self.line_file != -1 and file != self.line_file:
            raise ValueError("Trying to load different line than the one specified in the metafile! " + \
                             "This is not allowed. If you want to remove the line from the metafile, " + \
                             "please do DA.line_file = None.")
        if self.line_file == -1:
            # Remove existing line that was added manually
            self.line = None
        if file is None:
            if self.line_file is None:
                print("Warning: no file given and no file specified in the metafile! No line loaded.")
                return
            else:
                file = self.line_file
        with ProtectFile(file, 'r') as pf:
            line = json.load(pf)

        # Fix string keys in JSON: seeds are int
        if self.meta.nseeds > 0:
            if len(line.keys()) > self.meta.nseeds:
                raise ValueError("Line file not compatible with seeds! Expected a dict of lines with seeds as keys.")
            line = {int(seed): xt.Line.from_dict(l) for seed, l in line.items()}
        else:
            line = xt.Line.from_dict(line)
        self._line = line
        self.line_file = file



    # =================================================================
    # ================ Generation of intial conditions ================
    # =================================================================

    def _prepare_generation(self, normalised_emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        # Does survival already exist?
        if self.survival_data is not None:
            print("Warning: Initial conditions already exist! No generation done.")
            return

        # Avoid writing to the meta at every property
        self.meta._store_properties = False

        # If non-default values are specified, copy them to the metadata
        # In the grid generation further below, only the metadat values
        # should be used (and not the function ones)!
        if normalised_emittance is not None:
            self.update_emittance(normalised_emittance, update_surv=False)
        if self.normalised_emittance is None:
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


    def set_coordinates(self, *,
                        x=None, px=None, y=None, py=None, zeta=None, delta=None,
                        normalised_emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        """
        Let user provide initial coordinates for each plane.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        user_provided_coords = [i for i in [x,px, y, py, zeta, delta] if i is not None]

        # check that all provided lists have the same length
        assert len({len(i) for i in user_provided_coords}) == 1, 'Mismatch in length of provided lists'

        # replace all unused variables with zero
        x, px, y, py, zeta, delta = [ i if i is not None else 0 for i in [x,px, y, py, zeta, delta]]

        # Make all combinations
        if self.meta.nseeds > 0:
            self.meta.nseeds = nseeds
            seeds = np.arange(1,nseeds+1)
            x, px, y, py, zeta, delta, seeds = np.array(np.meshgrid(x, px, y, py, zeta, delta, seeds)).reshape(7,-1)

        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
            self._surv['seed'] = seeds.astype(int)

        self._surv['nturns'] = -1
        self._surv['x_norm_in'] = x
        self._surv['y_norm_in'] = y
        self._surv['px_norm_in'] = px
        self._surv['py_norm_in'] = py
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
        self.write_surv()
        self.meta.da_type = 'free'
        self.meta.da_dim = len(user_provided_coords)
        # self.meta.r_max = np.max(np.sqrt(x**2 + y**2))
        self.meta.npart = len(self._surv.index)
        self.meta._store()



    # Not allowed on parallel process
    def generate_initial_grid(self, *,
                                x_min, x_max, x_step=None, x_num=None,
                                y_min, y_max, y_step=None, y_num=None,
                                px_norm=0, py_norm=0, zeta=0, delta=0.00027,
                                normalised_emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        """Generate the initial conditions in a 2D X-Y grid.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        # Make the grid in xy
        def check_options(coord_min, coord_max, coord_step, coord_num, plane):
            if coord_step is None and coord_num is None:
                raise ValueError(f"Specify at least '{plane}_step' or '{plane}_num'.")
            elif coord_step is not None and coord_num is not None:
                raise ValueError(f"Use only one of '{plane}_step' and '{plane}_num', not both.")
            elif coord_step is not None:
                coord_num = floor( (coord_max-coord_min) / coord_step ) + 1
                coord_max = coord_min + (coord_num-1) * coord_step
            return coord_min, coord_max, coord_num

        x_min, x_max, x_num = check_options(x_min, x_max, x_step, x_num, 'x')
        y_min, y_max, y_num = check_options(y_min, y_max, y_step, y_num, 'y')

        x_space = np.linspace(x_min, x_max, x_num)
        y_space = np.linspace(y_min, y_max, y_num)

        # Make all combinations
        if self.meta.nseeds > 0:
            self.meta.nseeds = nseeds
            seeds = np.arange(1,nseeds+1)
            x, y, seeds = np.array(np.meshgrid(x_space, y_space, seeds)).reshape(3,-1)
        else:
            x, y = np.array(np.meshgrid(x_space, y_space)).reshape(2,-1)

        # Make dataframe
        self._surv = pd.DataFrame()
        if self.meta.nseeds > 0:
            self._surv['seed'] = seeds.astype(int)
        self._surv['ang_xy'] = np.arctan2(y,x)*180/np.pi
        self._surv['r_xy'] = np.sqrt(x**2 + y**2)
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
        self.write_surv()
        self.meta.da_type = 'grid'
        self.meta.da_dim = 2
        self.meta.r_max = np.max(np.sqrt(x**2 + y**2))
        self.meta.npart = len(self._surv.index)
        self.meta._store()



    # Not allowed on parallel process
    def generate_initial_radial(self, *, angles, r_min, r_max, r_step=None, r_num=None, ang_min=None, ang_max=None,
                                px_norm=0, py_norm=0, zeta=0, delta=0.00027,
                                normalised_emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        """Generate the initial conditions in a 2D polar grid.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        # Make the grid in r
        if r_step is None and r_num is None:
            raise ValueError("Specify at least 'r_step' or 'r_num'.")
        elif r_step is not None and r_num is not None:
            raise ValueError("Use only one of 'r_step' and 'r_num', not both.")
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
        self.write_surv()
        self.meta.da_type = 'radial'
        self.meta.da_dim = 2
        self.meta.r_max = r_max
        self.meta.npart = len(self._surv.index)
        self.meta._store()



    # Not allowed on parallel process
    def generate_random_initial(self, *, num_part=1000, r_max=25, px_norm=0, py_norm=0, zeta=0, delta=0.00027, ang_min=None,
                                ang_max=None, normalised_emittance=None, nseeds=None, pairs_shift=0, pairs_shift_var=None):
        """Generate the initial conditions in a 2D random grid.
        """

        self._prepare_generation(normalised_emittance, nseeds, pairs_shift, pairs_shift_var)

        # Make the data
        rng = default_rng()
        ang_min = 0 if ang_min is None else ang_min
        ang_max = 90 if ang_max is None else ang_max
        if self.meta.nseeds > 0:
            r = rng.uniform(low=0, high=r_max**2, size=num_part*self.meta.nseeds)
            th = rng.uniform(low=ang_min, high=ang_max*np.pi/180, size=num_part*self.meta.nseeds)
            seeds = np.repeat(np.arange(1,self.meta.nseeds+1), num_part)
        else:
            r = rng.uniform(low=0, high=r_max**2, size=num_part)
            th = rng.uniform(low=ang_min, high=ang_max*np.pi/180, size=num_part)
        r = np.sqrt(r)
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
        self.write_surv()
        self.meta.da_type = 'monte_carlo'
        self.meta.da_dim = 2
        self.meta.r_max = r_max
        self.meta.npart = len(self._surv.index)
        self.meta._store()



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
            if self.meta.min_turns is not None and self.meta.min_turns != min_turns:
                # TODO: This should only be checked if we already re-sampled.
                # There is no harm by changing min_turns after the initial run
                print(f"Warning: 'min_turns' can be set only once (and is already set to {self.meta.min_turns}). "
                      + "Ignored the new value.")

        # Get existing results
        if self.survival_data is None:
            raise ValueError("No survival data found!")
        data = self.survival_data
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

        print("Generating samples... ", end='')
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
        self.write_surv()



    # =================================================================
    # =================== create line from MAD-X ======================
    # =================================================================

    # TODO: can we get particle mass from mask??? Particle type??
    # Allowed on parallel process
    def build_line_from_madx(self, sequence, *, file=None, apertures=False, errors=True, \
                             mass=xp.PROTON_MASS_EV, store_line=True, seeds=None, run_all_seeds=False):
        # seeds=None  ->  find seed automatically (on parallel process)
        from cpymad.madx import Madx
        if file is None: 
            if self.madx_file is None:
                raise ValueError("No MAD-X file specified in arguments nor in metadata. Cannot build line.")
        else:
            if self.madx_file is None:
                self.madx_file = file
            elif self.madx_file != file:
                self.madx_file = file
                print("Warning: MAD-X file specified in arguments does not match with metadata. " + \
                      "Used the former and updated the metadata.")

        if store_line and self.line_file is None:
            self.line_file = self.madx_file.parent / (self.madx_file.stem + '.line.json')

        # Find which seeds to run, and initialise line file if it is to be stored
        if self.meta.nseeds == 0:
            seeds = [ None ]
        else:
            self._line = {}
            if run_all_seeds:
                if seeds is not None:
                    raise ValueError("Cannot have run_all_seeds=True and seeds!=None. Choose one.")
                seeds = range(1, self.meta.nseeds+1)
                print("Calculating the line from MAD-X for all available seeds. Make sure this is not ran on " + \
                      "a parallel process, as results might be unpredictable and probably wrong.")
                if store_line:
                    if self.line_file.exists():
                        print(f"Warning: line_file {self.line_file} exists and is being overwritten.")
                    with ProtectFile(self.line_file, 'w') as pf:
                        data = {str(seed): 'Running MAD-X' for seed in seeds}
                        json.dump(data, pf, cls=xo.JEncoder, indent=True)
            else:
                if seeds is not None:
                    if not hasattr(seeds, '__iter__'):
                        seeds = [ seeds ]
                elif not store_line:
                    raise ValueError("Cannot determine seed automatically (run_all_seeds=False and seed=None) " + \
                                     "if store_line=False, as the line file is needed to find seeds for which " + \
                                     "the MAD-X calculation isn't yet performed.")
                if store_line:
                    # Hack to make it thread-safe
                    created = False
                    if not self.line_file.exists():
                        try:
                            with ProtectFile(self.line_file, 'x') as fid:
                                if seeds is None:
                                    seeds = [ 1 ]
                                data = {str(seed): 'Running MAD-X' for seed in seeds}
                                json.dump(data, fid, cls=xo.JEncoder, indent=True)
                            created = True
                        except FileExistsError:
                            pass
                    if not created:
                        with ProtectFile(self.line_file, 'r+') as pf:
                            data = json.load(pf)
                            if seeds is None:
                                n_existing = len(data.keys())
                                if n_existing == self.meta.nseeds:
                                    # TODO: implement system to rerun failed MAD-X
#                                     to_redo = False
#                                     for seed, val in data.items():
#                                         if val == 'MAD-X failed':
#                                             seeds = [ int(seed) ]
#                                             to_redo = True
#                                             break
#                                     if not to_redo:
                                    print("File already has a line for every seed. Nothing to be done.")
                                    return
                                else:
                                    seeds = [ n_existing + 1 ]
                            for seed in seeds:
                                if str(seed) in data.keys():
                                    print(f"Warning: line_file {self.line_file} already contained a result for seed {seed}. " + \
                                           "This is being overwritten.")
                                data[str(seed)] = 'Running MAD-X'
                            pf.truncate(0)  # Delete file contents (to avoid appending)
                            pf.seek(0)      # Move file pointer to start of file
                            json.dump(data, pf, cls=xo.JEncoder, indent=True)    
                print(f"Processing seed{'s' if len(seeds) > 1 else ''} {seeds[0] if len(seeds) == 1 else seeds}.")

        # Run MAD-X
        energy = []
        for seed in seeds:
            with tempfile.TemporaryDirectory() as tmpdir:
                if seed is None:
                    madin = self.madx_file
                    madout = self.madx_file.parent / (self.madx_file.stem + '.madx.out')
                else:
                    madin = Path(tmpdir) / self.madx_file.name
                    madout = self.madx_file.parent / (self.madx_file.stem + '.' + str(seed) + '.madx.out')
                    with ProtectFile(self.madx_file, 'r') as fin:
                        data = fin.read()
                        with ProtectFile(madin, 'w') as fout:
                            fout.write(data.replace('%SEEDRAN', str(seed)))
                with ProtectFile(madout, 'w') as pf:
                    print(f"Running MAD-X in {tmpdir}")
                    mad = Madx(stdout=pf)
                    mad.chdir(tmpdir)
                    mad.call(madin.as_posix())
                    # TODO: check if MAD-X failed: if so, write 'MAD-X failed' into line_file
            line = xt.Line.from_madx_sequence(mad.sequence[sequence], apply_madx_errors=errors, \
                                              install_apertures=apertures)
            line.particle_ref = xp.Particles(mass0=mass, gamma0=mad.sequence[sequence].beam.gamma)
            energy = [ *energy, line.particle_ref.p0c[0] ]

            # Save result
            if seed is None:
                self._line = line
                if store_line:
                    with ProtectFile(self.line_file, 'w') as pf:
                        json.dump(self.line.to_dict(), pf, cls=xo.JEncoder, indent=True)
            else:
                if self._line is None:
                    self._line = {}
                self._line[seed] = line
                if store_line:
                    with ProtectFile(self.line_file, 'r+') as pf:
                        data = json.load(pf)
                        data[str(seed)] = line.to_dict()
                        pf.truncate(0)  # Delete file contents (to avoid appending)
                        pf.seek(0)      # Move file pointer to start of file
                        json.dump(data, pf, cls=xo.JEncoder, indent=True)

        # Check energy
        if self.meta.energy is not None:
            energy = [ *energy, self.meta.energy ]
        if len(np.unique([ round(e, 4) for e in energy])) != 1:
            raise ValueError(f"The lines for the different seeds have different energies: {energy}!")
        if self.meta.energy is None:
            self.meta.energy = energy[0]



    # =================================================================
    # ==================== Xtrack tracking jobs =======================
    # =================================================================

    # Allowed on parallel process
    def track_job(self, *,  npart=None, logging=True, force_single_seed_per_job=None):
        if self.line is None:
            if self.meta.line_file is not None and self.meta.line_file != -1 and self.meta.line_file.exists():
                self.load_line_from_file()
            else:
                raise Exception("No line loaded nor found on file!")

        # Create a job: get job ID and start logging
        part_ids, seeds, flag = self._create_job(npart, logging, force_single_seed_per_job)
        if flag != 0:
            return
        job_id = str(self._active_job)

        # Build tracker(s) if not yet done
        if self.meta.nseeds == 0:
            if self.line.tracker is None:
                print("Building tracker.")
                self.line.build_tracker()
        else:
            if 0 in self.line.keys():
                # Line file dict is 0-indexed
                seeds = [ seed-1 for seed in seeds ]
            for seed in seeds:
                if self.line[seed].tracker is None:
                    print(f"Building tracker for seed {seed}.")
                    self.line[seed].build_tracker()

        # TODO: DOES NOT WORK WITH SEEDS
        # Create initial particles
        x_norm  = self._surv['x_norm_in'].to_numpy()
        y_norm  = self._surv['y_norm_in'].to_numpy()
        px_norm = self._surv['px_norm_in'].to_numpy()
        py_norm = self._surv['py_norm_in'].to_numpy()
        zeta    = self._surv['zeta_in'].to_numpy()
        delta   = self._surv['delta_in'].to_numpy()

        context = self.line.tracker._buffer.context
        part = xp.build_particles(_context=context,
                          tracker=self.line.tracker,
                          x_norm=x_norm, y_norm=y_norm, px_norm=px_norm, py_norm=py_norm, zeta=zeta, delta=delta,
                          nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y
                         )
        # Track
        self._append_job_log('output', datetime.datetime.now().isoformat() + '  Start tracking job ' + str(job_id) + '.', logging=logging)
        self.line.tracker.track(particles=part, num_turns=self.meta.max_turns)
        context.synchronize()
        self._append_job_log('output', datetime.datetime.now().isoformat() + '  Done tracking job ' + str(job_id) + '.', logging=logging)

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

        if self.surv_exists():
            with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                self.read_surv(pf)
                self._surv.loc[part_ids, 'finished'] = True
                self._surv.loc[part_ids, 'x_out'] = x_out
                self._surv.loc[part_ids, 'y_out'] = y_out
                self._surv.loc[part_ids, 'nturns'] = survturns.astype(np.int64)
                self._surv.loc[part_ids, 'px_out'] = px_out
                self._surv.loc[part_ids, 'py_out'] = py_out
                self._surv.loc[part_ids, 'zeta_out'] = zeta_out
                self._surv.loc[part_ids, 'delta_out'] = delta_out
                self._surv.loc[part_ids, 's_out'] = s_out
                self._surv.loc[part_ids, 'state'] = state
                self.write_surv(pf)
        else:
            self._surv.loc[part_ids, 'finished'] = True
            self._surv.loc[part_ids, 'x_out'] = x_out
            self._surv.loc[part_ids, 'y_out'] = y_out
            self._surv.loc[part_ids, 'nturns'] = survturns.astype(np.int64)
            self._surv.loc[part_ids, 'px_out'] = px_out
            self._surv.loc[part_ids, 'py_out'] = py_out
            self._surv.loc[part_ids, 'zeta_out'] = zeta_out
            self._surv.loc[part_ids, 'delta_out'] = delta_out
            self._surv.loc[part_ids, 's_out'] = s_out

        self._update_job_log({
            'finished_time': datetime.datetime.now().isoformat(),
            'status': 'Finished'
        }, logging=logging)


    # NOT allowed on parallel process!
    def resubmit_unfinished(self):
        if self.surv_exists():
            with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                self.read_surv(pf)
                mask = self._surv['finished'] == False
                self._surv.loc[mask, 'submitted'] = False
                self.write_surv(pf)
        else:
            print("No survival file found! Nothing done.")

    # TODO: need to test
    def reset_survival_output(self, mask=None, mask_by=None, mask_func=None):
        if mask is None:
            if mask_by is None:
                if mask_func is None:
                    mask = slice(None,None,None)
                else:
                    raise ValueError("Cannot use 'mask_func' without 'mask_by'!")
            else:
                if mask_func is None:
                    raise ValueError("Need to use 'mask_func' when using 'mask_by'!")
        elif mask_by is not None or mask_func is not None:
            raise ValueError("Have to choose between using 'mask=..', or 'mask_by=..' and 'mask_func=..'!")

        def _reset_surv(mask):
            self._surv.loc[mask, 'x_out'] = -1
            self._surv.loc[mask, 'y_out'] = -1
            self._surv.loc[mask, 'px_out'] = -1
            self._surv.loc[mask, 'py_out'] = -1
            self._surv.loc[mask, 'zeta_out'] = -1
            self._surv.loc[mask, 'delta_out'] = -1
            self._surv.loc[mask, 's_out'] = -1
            self._surv.loc[mask, 'nturns'] = -1
            self._surv.loc[mask, 'state'] = 1
            self._surv.loc[mask, 'submitted'] = False
            self._surv.loc[mask, 'finished'] = False

        if self.surv_exists():
            with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                self.read_surv(pf)
                if mask is None:
                    mask = mask_func(self._surv[mask_by])
                _reset_surv(mask)
                self.write_surv(pf)
        elif self._surv is not None:
            if mask is None:
                mask = mask_func(self._surv[mask_by])
            _reset_surv(mask)


    # =================================================================
    # ========================= Calculate DA ==========================
    # =================================================================


    # Not allowed on parallel process
    def calculate_da(self):
        if self.da_type == 'radial':
            pass
        elif self.da_type == 'grid':
            pass
        elif self.da_type in ['monte_carlo', 'free']:
            pass # ML


    # =================================================================
    # ==================== Manage tracking jobs =======================
    # =================================================================

    # Allowed on parallel process
    def _create_job(self, npart=None, logging=True, force_single_seed_per_job=None):
        def _get_seeds_and_stuff(npart, logging):
            mask = self._surv['submitted'] == False
            if npart is None:
                this_part_ids = self._surv[mask].index
            else:
                this_part_ids = self._surv[mask].index[:npart]
            if self.meta.nseeds > 0:
                df = self._surv.loc[this_part_ids]
                seeds = np.unique(df['seed'])
                if force_single_seed_per_job:
                    # Only take jobs from one seed
                    seeds = seeds[:1]
                    mask = df['seed'] == seeds[0]
                    this_part_ids = df.loc[mask].index
            else:
                seeds = None
            # Quit the job if no particles need to be submitted
            if len(this_part_ids) == 0:
                print("No more particles to submit! Exiting...")
                if logging:
                    # The submissions log for this job will only have a status field
                    self.meta.update_submissions(self._active_job, {'status': 'No submission needed.'})
                return this_part_ids, seeds, -1
            # Otherwise, flag the particles as submitted, before releasing the file again
            self._surv.loc[this_part_ids, 'submitted'] = True
            return this_part_ids, seeds, 0


        # Get job ID
        self._active_job = self.meta.new_submission_id() if logging else None

        # Initialise force_single_seed_per_job
        if npart is None:
            # If tracking all particles, default to all seeds as well
            if force_single_seed_per_job is None:
                force_single_seed_per_job = False
            txt = ' over all seeds' if self.meta.nseeds > 0 and not force_single_seed_per_job else ''
            print(f"Tracking all available particles{txt}. Make sure this is not ran on a parallel process, " + \
                   "as results might be unpredictable and probably wrong.")
        else:
            force_single_seed_per_job = True if force_single_seed_per_job is None else force_single_seed_per_job

        # Get available particles
        if self.surv_exists():
            with ProtectFile(self.meta.surv_file, 'r+b', wait=_db_access_wait_time) as pf:
                # Get the first npart particle IDs that are not yet submitted
                # TODO: this can probably be optimised by only reading last column
                self.read_surv(pf)
                this_part_ids, seeds, flag = _get_seeds_and_stuff(npart, logging)
                if flag == 0:
                    self.write_surv(pf)
        else:
            this_part_ids, seeds, flag = _get_seeds_and_stuff(npart, logging)

        # Prepare job
        if flag == 0:
            # Reduce dataframe to only those particles in this job
            self._surv = self._surv.loc[this_part_ids]

            # Submission info
            if logging:
                self._active_job_log = {
                        'submission_time': datetime.datetime.now().isoformat(),
                        'finished_time':   0,
                        'status':          'Running',
                        'tracking_turns':  self.meta.max_turns,
                        'particle_ids':    '[' + ', '.join([str(pid) for pid in this_part_ids]) + ']',
                        'seeds':           [ int(seed) for seed in seeds ] if seeds is not None else None,
                        'warnings':        [],
                        'output':          [],
                }

        return this_part_ids, seeds, flag



    # Allowed on parallel process
    def _update_job_log(self, update, logging=True):
        if logging:
            self._active_job_log.update(update)
            self.meta.update_submissions(self._active_job, self._active_job_log)

    # Allowed on parallel process
    def _append_job_log(self, key, update, logging=True):
        if logging:
            self._active_job_log[key].append(update)
            self.meta.update_submissions(self._active_job, self._active_job_log)

    # Allowed on parallel process
    def _warn_job(self, warntext, logging=True):
        self._append_job_log('warnings', warntext, logging=logging)
        print(warntext)

    # Allowed on parallel process
    def _fail_job(self, failtext, logging=True):
        self._update_job_log({
            'finished_time': datetime.datetime.now().isoformat(),
            'status': 'Failed: ' + failtext
        }, logging=logging)
        raise Exception(failtext)



    # =================================================================
    # ======================= Database handling =======================
    # =================================================================

    def surv_exists(self):
        return self.meta._use_files and self.meta.surv_file.exists()

    def read_surv(self, pf=None):
        if self.surv_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.surv_file, 'rb', wait=_db_access_wait_time) as pf:
                        self._surv = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._surv = pd.read_parquet(pf, engine="pyarrow")
            else:
                raise NotImplementedError
        else:
            return None

    def write_surv(self, pf=None):
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.surv_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._surv.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._surv.to_parquet(pf, index=True, engine="pyarrow")
            else:
                raise NotImplementedError

    def da_exists(self):
        return self.meta._use_files and self.meta.da_file.exists()

    def read_da(self, pf=None):
        if self.da_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_file, 'rb', wait=_db_access_wait_time) as pf:
                        self._da = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._da = pd.read_parquet(pf, engine="pyarrow")
            else:
                raise NotImplementedError
        else:
            return None

    def write_da(self, pf=None):
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._da.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._da.to_parquet(pf, index=True, engine="pyarrow")
            else:
                raise NotImplementedError

    def da_evol_exists(self):
        return self.meta._use_files and self.meta.da_evol_file.exists()

    def read_da_evol(self, pf=None):
        if self.da_evol_exists():
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_evol_file, 'rb', wait=_db_access_wait_time) as pf:
                        self._da_evol = pd.read_parquet(pf, engine="pyarrow")
                else:
                    self._da_evol = pd.read_parquet(pf, engine="pyarrow")
            else:
                raise NotImplementedError
        else:
            return None

    def write_da_evol(self, pf=None):
        if self.meta._use_files and not self.meta._read_only:
            if self.meta.db_extension=='parquet':
                if pf is None:
                    with ProtectFile(self.meta.da_evol_file, 'wb', wait=_db_access_wait_time) as pf:
                        self._da_evol.to_parquet(pf, index=True, engine="pyarrow")
                else:
                    pf.truncate(0)  # Delete file contents (to avoid appending)
                    pf.seek(0)      # Move file pointer to start of file
                    self._da_evol.to_parquet(pf, index=True, engine="pyarrow")
            else:
                raise NotImplementedError

    # =================================================================
    # ============================ Plot DA ============================
    # =================================================================
    
    
    # =================================================================
 
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
    



