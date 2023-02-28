import numbers
from pathlib import Path
import json

from .protectfile import ProtectFile


# The DAMeta class stores the info of the DA
# The difference with the DA class is that the latter is a single instance, potentially
# of many in parallel.
# Is this needed?

# Developers: if a new metadata field is added, the following steps have to be implemented:
#    - description in docstring
#    - added to _fields
#    - give default in DAMetaData._defaults
#    - potential initialisation in DA
#    - @property setter and getter

# TODO: missing particle selection ...   =>  ?

def regenerate_meta_file(name, **kwargs):
    """Function to manually regenerate the *.meta.json file, in case it got corrupted or deleted.
    
    """
    meta = _DAMetaData(name, use_files=False)
    if meta.meta_file.exists():
        print("Warning: metadata file already exists! Not regenerated.")
    else:
        # Initialise fields by kwargs
        for field in _DAMetaData._defaults:
            setattr(meta, '_' + field, kwargs.get(field, _DAMetaData._defaults[field]) )
        meta._store()
        return meta


class _DAMetaData:
    """Collects all info of a DA study, and keeps the .meta.json file in sync.
    
    Attributes
    ----------
    name : str
        Name of the study. All files will have this name as stem.
    path : pathlib.Path
        Path to the study folder. Different studies are allowed in the same
        folder, as long as they have different names.
    da_type : str
        The type of DA, representing how the initial conditions are
        generated. Possible types are:
        radial      : a polar grid , expressed by one amplitude and
                      (potentially multiple) angles.
        grid        : a rectangular grid.
        monte_carlo : random sampling, where the DA border region is
                      recognised by an ML model.
        free        : any form of initial conditions, where the DA border
                      region is recognised by an ML model.
    da_dim : int
        The dimension of the DA. This is the number of dimensions over
        which the initial conditions are varied.
    nemitt_x : float
        The horizontal emittance used to calculated the beam size which
        is used to calculate the initial conditions.
    nemitt_y : float
        The vertical emittance used to calculated the beam size which
        is used to calculate the initial conditions.
    r_max : float
        The maximum radius to generate initial particles. This is set
        automatically by the generation functions.
    min_turns : int
        The minimum number of turns to use for DA recognition.
    max_turns : int
        The maximum number of turns to track.
    npart : int
        The number of particles to track.
    energy : float
        The energy of the beam.
    nseeds : int
        The number of seeds used in the DA. If no seeds are used, this
        is equal to zero.
    pairs_shift : float
        The separation between two particles in a pair. If no pairs are
        used, this is equal to zero.
    pairs_shift_var : str
        The variable upon which to make the shift between pairs. This can
        be 'angle', 'r', 'x', 'y', 'px', 'py', 'zeta', or 'delta'.
    s_start : float
        The longitudinal position of the initial conditions in the lattice.
    meta_file : pathlib.Path
        Path to the metadata file (*.meta.json)
    madx_file : pathlib.Path
        Path to the madx file (*.madx)
    line_file : pathlib.Path
        Path to the xtrack line file (*.line.json)
    db_extension : str
        Type of file to use for the survival DataFrames. Possible extensions are:
        parquet     : the Apache database format (need pyarrow installed).
        csv         : the CSV file format. Not yet supported.
        db          : the SQLite database format. Not yet supported.
        hdfs        : the HDFS database format. Not yet supported.
    surv_file : pathlib.Path
        Path to the survival file (*.surv.parquet)
    da_file : pathlib.Path
        Path to the da file (*.da.parquet)
    da_evol_file : pathlib.Path
        Path to the da evolution file (*.da_evol.parquet)
    submissions : list
        A log with info about the submitted jobs. A new job ID can be
        generated with new_submission_id, and the log can be updated with
        update_submissions.
    """

    # Class Attributes
    # ----------------
    #
    # _fields is (only) used to define the order of fields in the json
    # _path_fields is used to list those that require a special treatment:
    #      They need to have a .to_posix() call before storing in the json
    # _auto_fields are calculated automatically and do not need to be read in
    
    _fields = ['name','path','da_type','da_dim','nemitt_x','nemitt_y','min_turns','max_turns','npart','energy','nseeds',\
               'r_max','pairs_shift','pairs_shift_var','s_start','meta_file','madx_file','line_file','db_extension',\
               'surv_file','da_file','da_evol_file','submissions']
    _path_fields = ['path','meta_file','madx_file','line_file','surv_file','da_file','da_evol_file']
    _auto_fields = ['name','path','meta_file','surv_file','da_file','da_evol_file']
    # used to specify the accepted DA types
    _da_types=['radial', 'grid', 'monte_carlo', 'free']
    # used to specify the accepted file formats for the survival dataframe
    _db_extensions=['parquet', 'db', 'hdfs']

    _defaults = {
        'da_type':         None,
        'da_dim':          None,
        'nemitt_x':        None,
        'nemitt_y':        None,
        'min_turns':       None,
        'max_turns':       None,
        'npart':           None,
        'r_max':           None,
        'energy':          None,
        'nseeds':          0,
        'pairs_shift':     0,
        'pairs_shift_var': None,
        's_start':         0,
        'submissions':     {},
        'line_file':       None,
        'madx_file':       None,
        'db_extension':    'parquet'
    } #| { field: None for field in _optional_fields}


    def __init__(self, name, *, path=Path.cwd(), use_files=False, read_only=False, parallel=False):
        # Remove .meta.json suffix if passed with filename
        if name.split('.')[-2:] == ['meta', 'json']:
            name = name[:-10]
        self._name             = name
        if parallel and not use_files:
            use_files = True
        self._parallel         = parallel
        self._use_files        = use_files
        self.path              = path
        self._store_properties = True

        # Initialise defaults
        for field in self._defaults:
            setattr(self, '_' + field, self._defaults[field])
        self._new = True
        if read_only and not use_files:
            read_only = False
        self._read_only = read_only
            

        if use_files:
            if self.meta_file.exists():
                print(f"Loading existing DA object ({self.name} in {self.path}).")
                self._read()
                self._new = False
            else:
                if self.surv_file.exists() or self.da_file.exists() or self.da_evol_file.exists():
                     raise ValueError("Tried to create new DA object, but some parquet files already exist!\n" \
                                     + "If you tried to load an existing DA object, make sure to keep the *.meta.json " \
                                     + "file in the same folder as the parquet files, or regenerate the metadata file " \
                                     + "manually with xdyna.regenerate_meta_file(). Or, if the parquet files are old/wrong, " \
                                     + "just delete them.")
                if parallel:
                    raise ValueError("Cannot create a new DA object on a parallel process!")
                if read_only:
                    raise ValueError("Specified read_only=True but no files found!")
                print("Creating new DA object.")
                self._store()



    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path if self._use_files else None

    @path.setter
    def path(self, path):
        self._path = Path(path).resolve() if self._use_files else None

    @property
    def meta_file(self):
        return Path(self.path, self.name + '.meta.json').resolve() if self._use_files else None

    @property
    def madx_file(self):
        return self._madx_file

    @madx_file.setter
    def madx_file(self, madx_file):
        madx_file = Path(madx_file).resolve()
        self._set_property('madx_file', madx_file)

    @property
    def line_file(self):
        return self._line_file

    @line_file.setter
    def line_file(self, line_file):
        if line_file != -1:
            line_file = Path(line_file).resolve()
        self._set_property('line_file', line_file)

    @property
    def surv_file(self):
        return Path(self.path, self.name + '.surv.' + self.db_extension).resolve() if self._use_files else None

    @property
    def da_file(self):
        return Path(self.path, self.name + '.da.' + self.db_extension).resolve() if self._use_files else None

    @property
    def da_evol_file(self):
        return Path(self.path, self.name + '.da_evol.' + self.db_extension).resolve() if self._use_files else None

    @property
    def db_extension(self):
        return self._db_extension if self._use_files else None

    @db_extension.setter
    def db_extension(self, db_extension):
        if self._use_files and db_extension != self.db_extension:
            if self.surv_file.exists() or self.da_file.exists() or self.da_evol_file.exists():
                raise NotImplementedError("DataFrame currently in different format! Need to translate..")
            if not db_extension in self._db_extensions:
                raise ValueError(f"The variable db_extension should be one of {', '.join(self._db_extensions)}!")
            if db_extension =='db' or db_extension=='hdfs':
                raise NotImplementedError
            self._set_property('db_extension', db_extension)

    @property
    def da_type(self):
        return self._da_type

    @da_type.setter
    def da_type(self, da_type):
        if not da_type in self._da_types:
            raise ValueError(f"The variable da_type should be one of {', '.join(self._da_types)}!")
        self._set_property('da_type', da_type)

    @property
    def da_dim(self):
        return self._da_dim

    @da_dim.setter
    def da_dim(self, da_dim):
        if not isinstance(da_dim, numbers.Number) or da_dim < 2 or da_dim > 6:
            raise ValueError(f"The variable da_dim should be a number between 2 and 6!")
        self._set_property('da_dim', round(da_dim))

    @property
    def nemitt_x(self):
        return self._nemitt_x

    @nemitt_x.setter
    def nemitt_x(self, nemitt_x):
        if not isinstance(nemitt_x, numbers.Number):
            raise ValueError(f"The emittance should be a number!")
        if nemitt_x <= 0:
            raise ValueError(f"The emittance has to be larger than zero!")
        self._set_property('nemitt_x', nemitt_x)

    @property
    def nemitt_y(self):
        return self._nemitt_y

    @nemitt_y.setter
    def nemitt_y(self, nemitt_y):
        if not isinstance(nemitt_y, numbers.Number):
            raise ValueError(f"The emittance should be a number!")
        if nemitt_y <= 0:
            raise ValueError(f"The emittance has to be larger than zero!")
        self._set_property('nemitt_y', nemitt_y)

    @property
    def r_max(self):
        return self._r_max

    @r_max.setter
    def r_max(self, r_max):
        if not isinstance(r_max, numbers.Number) and r_max is not None:
            raise ValueError(f"The property r_max should be a number or None!")
        if r_max is not None and r_max <= 0:
            raise ValueError(f"The property r_max has to be larger than zero!")
        self._set_property('r_max', r_max)

    @property
    def min_turns(self):
        return self._min_turns

    @min_turns.setter
    def min_turns(self, min_turns):
        if not isinstance(min_turns, numbers.Number) and min_turns is not None:
            raise ValueError(f"The value of min_turns should be a number or None!")
        if min_turns == 0:
            min_turns = None
        if min_turns is not None and min_turns <= 0:
            raise ValueError(f"The property min_turns has to be larger than zero!")
        min_turns = round(min_turns) if min_turns is not None else None
        self._set_property('min_turns', min_turns)
        
    @property
    def max_turns(self):
        return self._max_turns

    @max_turns.setter
    def max_turns(self, max_turns):
        if not isinstance(max_turns, numbers.Number):
            raise ValueError(f"The value of max_turns should be a number!")
        if max_turns <= 0:
            raise ValueError(f"The property max_turns has to be larger than zero!")
        self._set_property('max_turns', round(max_turns))

    @property
    def npart(self):
        return self._npart

    @npart.setter
    def npart(self, npart):
        if not isinstance(npart, numbers.Number) and npart is not None:
            raise ValueError(f"The value of npart should be a number or None!")
        if npart is not None and npart <= 0:
            raise ValueError(f"The property npart has to be larger than zero!")
        npart = round(npart) if npart is not None else None
        self._set_property('npart', npart)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if not isinstance(energy, numbers.Number) and energy is not None:
            raise ValueError(f"The energy should be a number or None!")
        if energy is not None and energy <= 0:
            raise ValueError(f"The energy has to be larger than zero!")
        self._set_property('energy', energy)

    @property
    def nseeds(self):
        return self._nseeds

    @nseeds.setter
    def nseeds(self, nseeds):
        if not isinstance(nseeds, numbers.Number):
            raise ValueError(f"The number of seeds should be a number!")
        self._set_property('nseeds', round(nseeds))

    @property
    def pairs_shift(self):
        return self._pairs_shift

    @pairs_shift.setter
    def pairs_shift(self, pairs_shift):
        if not isinstance(pairs_shift, numbers.Number):
            raise ValueError(f"The variable pairs_shift should be a number!")
        self._set_property('pairs_shift', pairs_shift)

    @property
    def pairs_shift_var(self):
        return self._pairs_shift_var

    @pairs_shift_var.setter
    def pairs_shift_var(self, pairs_shift_var):
        accepted = ['r', 'angle', 'x', 'y', 'px', 'py', 'zeta', 'delta']
        if pairs_shift_var not in accepted:
            raise ValueError(f"The variable pairs_shift_var should be one of {', '.join(accepted)}!")
        self._set_property('pairs_shift_var', pairs_shift_var)

    @property
    def s_start(self):
        return self._s_start

    @s_start.setter
    def s_start(self, s_start):
        if not isinstance(s_start, numbers.Number):
            raise ValueError(f"The variable s_start should be a number!")
        self._set_property('s_start', s_start)

    @property
    def submissions(self):
        return self._submissions if self._use_files else None

    # Allowed on parallel process.
    def new_submission_id(self):
        if not self._use_files or self._read_only:
            return None
        else:
            with ProtectFile(self.meta_file, 'r+', wait=0.005) as pf:
                meta = json.load(pf)
                new_id = len(meta['submissions'].keys())
                meta['submissions'][new_id] = {}
                pf.truncate(0)  # Delete file contents (to avoid appending)
                pf.seek(0)      # Move file pointer to start of file
                json.dump(meta, pf, indent=2, sort_keys=False)
                self._submissions = meta['submissions']
                return new_id

    # Allowed on parallel process (but only if each process updates only the log attached to its unique ID).
    # This will overwrite the value associated to submission_id in self._submissions with val.
    def update_submissions(self, submission_id, val):
        if self._use_files and not self._read_only:
            with ProtectFile(self.meta_file, 'r+', wait=0.005) as pf:
                meta = json.load(pf)
                meta['submissions'].update({str(submission_id): val})
                pf.truncate(0)  # Delete file contents (to avoid appending)
                pf.seek(0)      # Move file pointer to start of file
                json.dump(meta, pf, indent=2, sort_keys=False)
                self._submissions = meta['submissions']

    # Not allowed on parallel process!
    def _set_property(self, prop, val):
        if getattr(self, '_' + prop) != val:
            setattr(self, '_' + prop, val)
            if self._store_properties:
                self._check_not_changed_and_store(ignore=[prop])
    
    def _check_not_changed_and_store(self, ignore=[]):
        if self._use_files and not self._read_only:
            # Create dict of self fields, ignoring the field that is expected to change
            sortkeys = [ x for x in self._fields if x not in ignore ]
            thisdict = { key: getattr(self, key) for key in sortkeys }
            # Special treatment for paths: make them strings
            self._paths_to_strings(thisdict, ignore)
            # Load file
            with ProtectFile(self.meta_file, 'r+') as pf:
                meta = json.load(pf)
                meta = { key: meta[key] for key in sortkeys }
                # Compare
                if meta != thisdict:
                    raise Exception("The metadata file changed on disk!\n" \
                                   + "This is not supposed to happen, and probably means that one of the child processes " \
                                   + "tried to write to it (which is only allowed for the 'submissions' field).\n" \
                                   + "Please check your workflow.")
                else:
                    self._store(pf=pf)

    def _read(self):
        if self._use_files:
            # The _auto_fields (all paths) are calculated automatically at creation.
            # If those values in the file diverge from the actual ones, the file needs to be updated
            # (this happens e.g. if the study is moved to a different folder)
            need_to_store = False
            with ProtectFile(self.meta_file, 'r+') as pf:
                meta = json.load(pf)
                # clean up paths in meta
                for field in self._path_fields:
                    val = meta.get(field, None)
                    if val is not None:
                        meta[field] = Path(meta[field])
                    if field == 'line_file' and val == 'line manually added':
                        meta[field] = -1
                for field in self._fields:
                    if field in self._auto_fields and getattr(self, field)!=meta.get(field, None):
                        need_to_store = True
                    else:
                        # Default to None, in case of optional keys
                        val = meta.get(field, None)
                        setattr(self, '_' + field, val )
                if need_to_store:
                    # Special treatment for line:
                    # If its path is the same as the path variable in the file, then it is considered to be
                    # in the same folder als the study and its path will be updated as well. Otherwise it is considered
                    # to have an absolute path and it won't be updated
                    if self.line_file is not None \
                            and self.line_file.parent==Path(meta['path']) \
                            and Path(self.path, self._line_file.name).exists():
                        self._line_file = Path(self.path, self._line_file.name)
                    self._store(pf=pf)

    def _store(self, pf=None):
        self._store_properties = True
        if self._use_files and not self._read_only:
            meta = { key: getattr(self, key) for key in self._fields }
            self._paths_to_strings(meta)
            if pf is None:
                mode = 'r+' if self.meta_file.exists() else 'x+'
                with ProtectFile(self.meta_file, mode) as pf:
                    if mode == 'r+':
                        pf.truncate(0)  # Delete file contents (to avoid appending)
                        pf.seek(0)      # Move file pointer to start of file
                    json.dump(meta, pf, indent=2, sort_keys=False)
            else:
                pf.truncate(0)  # Delete file contents (to avoid appending)
                pf.seek(0)      # Move file pointer to start of file
                json.dump(meta, pf, indent=2, sort_keys=False)
    
    def _paths_to_strings(self, meta, ignore=[]):
        update_dict = {}
        for key in self._path_fields:
            if key not in ignore:
                if getattr(self,key) is not None:
                    if key == 'line_file' and getattr(self,key) == -1:
                        update_dict[key] = 'line manually added'
                    else:
                        update_dict[key] = getattr(self,key).as_posix()
                else:
                    update_dict[key] = None
        meta.update(update_dict)
