from time import process_time
import numpy as np
import pandas as pd
import scipy as sp


# pip install scikit-learn-intelex
# try:
#     from sklearnex import patch_sklearn #, unpatch_sklearn
#     patch_sklearn()
# except ImportError:
#     pass

# try:
#     # Rapids is scikit-learn on gpu's
#     # download via conda; see https://rapids.ai/start.html#get-rapids for info
#     from cuml.svm import SVC
# except ImportError:
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage import measure

from sklearn.utils.fixes import loguniform

from .geometry import in_polygon_2D


# TODO: catch if border does not have enough points (i.e. goes out of range) if minimum is taken
# too small. Solution is to increase Nmin, or add a wider range of points

class MLBorder:
    def __init__(self, x, y, turns_surv, *, at_turn, memory_threshold=1e9, cv=None, cv_matrix=None,
                 margin=0.2, min_samples_per_label=100, try_to_balance_input=True):
        self._turn = at_turn
        self._memory_threshold = memory_threshold
        self._input_time = 0
        self._fit_time = 0
        self._evaluation_time = 0
        self._cv_matrix = None # Needed to make set_input_data work correctly

        self.set_input_data(x, y, turns_surv, margin, min_samples_per_label, try_to_balance_input)

        # If a cv_matrix is given, it sets the number of CV splits and constructs a model
        self._cv_matrix = cv_matrix
        if cv_matrix is not None:
            self._cv = self._get_cv_from_matrix()
            if cv is not None and self._cv != cv:
                print(f"Warning: argument {cv=} in MLBorder constructor does not match cv from 'cv_matrix' "
                      + f"({self._cv}). Ignored the former.")
        elif cv is None:
            self._cv = 10
        else:
            self._cv = cv
        self._update_ml()



    @property
    def turn(self):
        return self._turn

    @property
    def volume(self):
        if self._volume is None:
            self._calculate_volume()
        return self._volume

    @property
    def border(self):
        if self._border_x is None or self._border_y is None:
            return None
        else:
            return np.array([ self._border_x, self._border_y ])

    @property
    def border_resolution(self):
        return self._border_res

    @property
    def memory_threshold(self):
        return self._memory_threshold

    @memory_threshold.setter
    def memory_threshold(self, val):
        self._memory_threshold = val
        if self._svc is not None:
            self._svc.cache_size = val / 1e6

    @property
    def time(self):
        return {
                'total':      self._input_time + self._fit_time + self._evaluation_time,
                'input':      self._input_time,
                'fit':        self._fit_time,
                'evaluation': self._evaluation_time
               }

    @property
    def input_data(self):
        return self._input_data

    @property
    def labels(self):
        return self._labels

    @property
    def n_samples(self):
        return None if self._labels is None else len(self._labels)

    @property
    def ml_possible(self):
        return self._ml_possible

    @property
    def ml_impossible_reason(self):
        return self._ml_impossible_reason

    @property
    def extra_sample_r_region(self):
        return self._extra_sample_r_region

    @property
    def cv(self):
        return self._cv

    @property
    def cv_matrix(self):
        return self._cv_matrix

    @property
    def predict(self):
        if self._svc is None:
            def _mock(*args, **kwargs):
                return None
            return _mock
        else:
            return self._svc_pipe.predict

    @property
    def ml_iterations(self):
        return len(self.cv_matrix.columns)

    @property
    def C(self):
        return self._C

    @property
    def gamma(self):
        return self._gamma

    @property
    def best_estimator_(self):
        return self._best_estimator_


    # TODO: improve scoring function, because now for some splits it clips at 1.
    def fit(self, iterations=50, *, cv=None):
        if not self.ml_possible:
            return
        start_time = process_time()
        previous_best = self.best_estimator_
        if cv is not None:
            if self.cv_matrix is None:
                self._cv = cv
            elif self._cv != cv:
                print(f"Warning: argument {cv=} in 'fit()' does not match cv from 'cv_matrix' ({self._cv}). "
                      + "Ignored the former.")
        svc = SVC(kernel="rbf", decision_function_shape='ovr', class_weight='balanced')
        svc.cache_size = self.memory_threshold / 1e6
        svc_pipe = make_pipeline(StandardScaler(), svc)
        svc_param_grid = {
            'svc__C':     loguniform(1e0, 1e5),
            'svc__gamma': loguniform(1e-2, 1e3),
         }
        clf = RandomizedSearchCV(svc_pipe, svc_param_grid, n_iter=iterations, n_jobs=-1, cv=self._cv)
        clf.fit(self.input_data, self.labels)
        self._update_cv_matrix(cv_result=clf.cv_results_, n_iter=iterations) 
        self._fit_time += process_time() - start_time

        # Fitting potentially invalidates a previous border
        if previous_best != self.best_estimator_:
            self._border_x = None
            self._border_y = None
            self._border_res = None
            self._volume = None


    def _get_cv_from_matrix(self):
        return sum([1 if 'score_' in str else 0 for str in self.cv_matrix.index]) - 4

    def _update_cv_matrix(self, cv_result, n_iter):
        
        # Number of CV splits in the results
        cv = sum([1 if 'split' in str else 0 for str in cv_result.keys()])
        
        # If we already have results from a previous scan, prepend them
        if self.cv_matrix is None:
            scores = [ 'score_' + str(i) for i in range(cv) ]
            cv_matrix = pd.DataFrame(index=[
                            'C', 'gamma', 'score_min', 'score_mean', 'score_max', 'score_std', *scores, 'fit_time'
                        ])
        else:
            # Check if previous runs were done with the same number of CV splits
            if cv != self._get_cv_from_matrix():
                raise ValueError(f"Current result used {cv} CV splits, while previous run was done "
                                 + f"with {previous_cv} CV splits. Cannot combine!")
            cv_matrix = self.cv_matrix

        last_id = len(cv_matrix.columns) - 1
        this_cv_matrix = {}
        for this_id in range(n_iter):
            cv_scores = { 'score_' + str(i): cv_result['split' + str(i) + '_test_score'][this_id] for i in range(cv) }
            this_cv_matrix[last_id + this_id + 1] = pd.Series({
                'C': cv_result['params'][this_id]['svc__C'],
                'gamma':      cv_result['params'][this_id]['svc__gamma'],
                'score_min':  min(cv_scores.values()),
                'score_mean': cv_result['mean_test_score'][this_id],
                'score_max':  max(cv_scores.values()),
                'score_std':  cv_result['std_test_score'][this_id],
                **cv_scores,
                'fit_time':   cv_result['mean_fit_time'][this_id] * cv,
            })

        self._cv_matrix = pd.concat([
                                cv_matrix,
                                pd.DataFrame(this_cv_matrix)
                            ], axis=1)
        self._update_ml()


    def _find_optimal_parameters(self):
        if self._cv_matrix is None:
            return
        # Best score is chosen by the best mean
        best = self._cv_matrix.loc['score_mean'].idxmax()
        self._best_estimator_ = best
        self._C = self._cv_matrix[best]['C']
        self._gamma = self._cv_matrix[best]['gamma']


    def _update_ml(self):
        if self._cv_matrix is None or not self.ml_possible:
            self._svc = None
            self._svc_pipe = None
            self._predict = None
            self._C = None
            self._gamma = None
            self._best_estimator_ = None
            self._border_x = None
            self._border_y = None
            self._border_res = None
            self._volume = None
        else:
            self._find_optimal_parameters()
            self._svc = SVC(kernel="rbf", decision_function_shape='ovr', class_weight='balanced')
            self._svc.C = self.C
            self._svc.gamma = self.gamma
            self._svc.cache_size = self.memory_threshold / 1e6
            self._svc_pipe = make_pipeline(StandardScaler(), self._svc)
            self._svc_pipe.fit(self.input_data, self.labels)



    def set_input_data(self, x, y, turns_surv, margin=0.2, min_samples_per_label=100, try_to_balance_input=True):
        start_time = process_time()
        # Setting new input data undoes previous fits
        if self.cv_matrix is not None:
            print("Warning: Removed previous existing cv_matrix!")
        self._cv_matrix = None
        self._update_ml()

        # Define the labels
        labels = np.zeros(turns_surv.shape)
        labels[turns_surv >= self.turn ] = 1
        labels[turns_surv  < self.turn ] = 0

        # Define the two categories
        n_1 = len(labels[labels==1])
        n_0 = len(labels[labels==0])
        r_0 = np.sqrt(x[labels==0]**2 + y[labels==0]**2)
        r_1 = np.sqrt(x[labels==1]**2 + y[labels==1]**2)

        # Check if we have enough samples in each category
        self._ml_possible = True
        self._ml_impossible_reason = None
        self._extra_sample_r_region = []
        if min_samples_per_label >= len(labels)/2:
            self._ml_possible = False
            self._ml_impossible_reason = -1
            self._extra_sample_r_region = [0, np.concatenate([r_0, r_1]).max()]
        if n_0 < min_samples_per_label:
            self._ml_possible = False
            self._ml_impossible_reason = 0
            self._extra_sample_r_region = [r_0.min()*(1-margin), r_0.max*(1+margin)]
        if n_1 < min_samples_per_label:
            self._ml_possible = False
            self._ml_impossible_reason = 1
            self._extra_sample_r_region = [0, r_1.max()*(1+margin)]
        if not self.ml_possible:
            self._input_data = None
            self._labels = None
            return

        # Try to balance the samples:
        mask = np.full_like(labels, True, dtype=bool)
        if try_to_balance_input:
            step = 0.01
            r_upper = r_0.max()
            r_lower = r_1.min()

            #   if n_0 > n_1:  by shrinking the 0-particles region radially, stepwise from outside inwards
            while n_0 > n_1*(1+margin) and r_upper >= r_1.max()*(1+margin) and n_0 > min_samples_per_label:
                mask = np.sqrt(x**2 + y**2) <= r_upper
                n_0 = len(labels[mask][labels[mask]==0])
                # The following needs to be last, to avoid the (unlikely) case where step would be larger
                # than the margin, and particles from the other category would be accidentally removed
                r_upper -= step

            #   if n_0 < n_1:  by shrinking the 1-particles region radially, stepwise from inside outwards
            while n_1 > n_0*(1+margin) and r_lower <= r_0.min()*(1+margin) and n_1 > min_samples_per_label:
                mask = np.sqrt(x**2 + y**2) >= r_lower
                n_1 = len(labels[mask][labels[mask]==1])
                # Dito as above
                r_lower += step

        # Mask the data and labels, and store it
        data = np.array([x,y]).T
        self._input_data = data[mask]
        self._labels = labels[mask]
        self._xmin = x[mask].min()
        self._xmax = x[mask].max()
        self._ymin = y[mask].min()
        self._ymax = y[mask].max()
        self._input_time += process_time() - start_time

#         # To limit a bit the number of samples, and
#         # to balance them a bit, 
#         # select a square region around the surviving points,
#         # with a (by default) 20% margin on each side
#         region_x = x[labels==1]
#         region_y = y[labels==1]
#         xmin, xmax = region_x.min(), region_x.max()
#         ymin, ymax = region_y.min(), region_y.max()
#         dx = xmax - xmin
#         dy = ymax - ymin
#         xmin -= margin*dx
#         xmax += margin*dx
#         ymin -= margin*dy
#         ymax += margin*dy

#         # Mask the data and labels, and store it
#         data = np.array([x,y]).T
#         mask = np.array([ xmin<=x<=xmax and ymin<=y<=ymax for x,y in data ])
#         self._input_data = data[mask]
#         self._labels = labels[mask]
#         self._ymin = ymin
#         self._ymax = ymax
#         self._xmin = xmin
#         self._xmax = xmax



    def evaluate(self, step_resolution=0.01):
        if self._svc is None:
            raise ValueError("ML model not yet fitted. Do this first.")
        # No need to evaluate if border exists and resolution is the same
        if self._border_x is not None and self._border_y is not None and self._border_res == step_resolution:
            return
        start_time = process_time()
        x_grid = np.arange(self._xmin, self._xmax, step_resolution)
        y_grid = np.arange(self._ymin, self._ymax, step_resolution)
        actual_x_max = x_grid[-1]
        actual_y_max = y_grid[-1]
        len_x = len(x_grid)
        len_y = len(y_grid)
        xx, yy = np.meshgrid(x_grid, y_grid)
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Then we find the contours of the data
        #
        # TODO: This whole hack to find the curve by using contour image analysis is not ideal.
        # We should look for smarter alternatives, or at least do the step_resolution iteratively
        contours = measure.find_contours(Z, 0.5)

        # The contours are not in coordinates but in list indices;
        # we need to retransform them into coordinates
        def x_converter(x):
            return self._xmin + (x/(len_x-1)*(actual_x_max - self._xmin))
        def y_converter(y):
            return self._ymin + (y/(len_y-1)*(actual_y_max - self._ymin))
        contours_converted = [
            np.array([ [x_converter(x), y_converter(y)] for y, x in contour ])
            for contour in contours
        ]

        # If several contours are present (i.e. islands), we choose the one containing the origin
        contour_found = False
        for contour in contours_converted:
            contour = contour.T
            if in_polygon_2D(0, 0, contour[0], contour[1]):
                if contour_found:
                    raise Exception("Several contours found around the origin! Please investigate.")
                else:
                    border_x = contour[0]
                    border_y = contour[1]
                    contour_found = True
        if not contour_found:
            raise Exception("No contour found around the origin! Please investigate.")

        self._border_x = border_x
        self._border_y = border_y
        self._evaluation_time += process_time() - start_time
        self._volume = None



    def _calculate_volume(self, int_points=None):
        if self._border_x is None or self._border_y is None:
            return
        border, _ = sp.interpolate.splprep(self.border, s=0, per=1)
        if int_points is None:
            int_points = 50*len(self._border_x)
        int_points = int(int_points)
        u = np.linspace(0, 1 - 1/int_points, int_points) + 1/2/int_points
        x_curv, _  = sp.interpolate.splev(u, border, der=0, ext=2)
        _, dy_curv = sp.interpolate.splev(u, border, der=1, ext=2)
        self._volume = (1/int_points*x_curv*dy_curv).sum()










