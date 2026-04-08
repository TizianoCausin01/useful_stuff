import sys, os, yaml
from datetime import datetime
import numpy as np
import h5py
import torch
import argparse
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score 
from numba import njit
from einops import reduce
from einops.einops import EinopsError


def print_wise(mex, rank=None):
    if rank == None:
        print(datetime.now().strftime("%H:%M:%S"), f"- {mex}", flush=True)
    else:
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"- rank {rank}",
            f"{mex}",
            flush=True,
        )


# EOF

"""
get_lagplot_checks
Does the sanity checks for the autocorrelation function.
1 - If the maximum lag required is larger than the datapts available raises an index error
2 - If the maximum lag required overcomes the number of minimum datapts asked to compute the average of a diagonal prints a warning
3 - If the matrix has nans (datapts with var=0 usually with norm=0) prints a warning  
"""
def get_lagplot_checks(corr_mat, max_lag, min_datapts):
    n_timepts = corr_mat.shape[0]
    if n_timepts <= max_lag:
        raise IndexError("The maximum lag is larger than the matrix itself")
    elif n_timepts < max_lag+min_datapts:
        print_wise(f"The number of datapoints used to compute extreme offsets is < than {min_datapts}")
    # end if n_timepts < max_lag:
    nan_mask = np.isnan(corr_mat)
    if np.any(nan_mask):
        print_wise(f"There are nans in corr_mat") #{np.where(nan_mask)}")
    # end if np.any(nan_mask):
# EOF


"""
nan_check
Checks if there are NaN vals in the correlation matrix
"""
def nan_check(corr_mat: np.ndarray):
    nan_mask = np.isnan(corr_mat)
    if np.any(nan_mask):
        print_wise(f"There are nans in corr_mat") #{np.where(nan_mask)}")
    # end if np.any(nan_mask):


"""
choose_summary_stat
To choose the summary statistics to summarize the diagonals of the autocorrelation matrix
"""
def choose_summary_stat(summary_stat: str):
    if summary_stat == 'mean':
        stat = np.nanmean
    elif summary_stat == 'median':
        stat = np.nanmedian
    else:
        raise ValueError("summary_stat must be 'mean' or 'median'")
    # end if summary_stat == 'mean':
    return stat


"""
autocorr_mat
Correlates one time-series matrix to itself or one another to yield a timepts x timepts matrix.
INPUT:
    - data: np.ndarray(float) -> (features x time_pts)
    - data2: np.ndarray(float) -> (features x time_pts) in case of cross-correlation

OUTPUT:
    - corr_mat: np.ndarray(float) -> (tpts x tpts) the matrix of cross-correlation
"""
def autocorr_mat(data, data2=None, metric='correlation'):
    if data2 is None:
        corr_mat = np.corrcoef(data, rowvar=False)
    else:
        if metric == 'correlation':
            d1_shape = data.shape
            d2_shape = data2.shape
            corr_mat = np.corrcoef(data, data2, rowvar=False) 
            corr_mat = corr_mat[:d1_shape[1], d2_shape[1]:]
        else:
            corr_mat = pairwise_distances(data.T, data2.T, metric=metric)
        # end if metric == 'correlation':
    # end if data2 is None:
    return corr_mat
# EOF


"""
create_RDM
Creates an RDM with a specific distance metric and then indexes it with the triu method.
INPUT:
- data: np.array (features x datapoints) -> the data matrix
- metric: str or custom function -> the distance metric
OUTPUT:
- RDM_vec: np.array ([1/2 *(datapoints^2 - datapoints)],) -> the upper triangular entries of the RDM (indexed in a row-major order, excluding diagonal)
                                                            to go back to the full matrix, it's just squareform(RDM_vec)
"""
def create_RDM(data, metric='correlation'):
    if data.shape[1] == 1:
        raise IndexError('Cannot compute RDM with only 1 trial')
    # end if data.shape[1] == 1:
    if metric == 'correlation':
        RDM = 1 - np.corrcoef(data, rowvar=False)
        RDM_vec = index_gram(RDM)
    elif metric == 'cosine':
        RDM_vec = cosine_sim(data)
    elif metric == 'cosine_cnt':
        data = mean_centering(data)
        RDM_vec = cosine_sim(data)
    elif metric == 'magnitude_diff':
        RDM_vec = magnitude_diff(data)
    else:
        RDM_vec = pdist(data.T, metric=metric)
    # end if metric == 'pearson':
    return RDM_vec
# EOF


def mean_centering(x: np.ndarray, axis=1):
    x_cnt = x - x.mean(axis=axis, keepdims=True)
    return x_cnt
# EOF

"""
spearman
Computes the spearman's rank correlation coefficient between two vectors.
the first argsort treats the position in the matrix as rank and the index as the position in the previous matrix. I.e. it gives the indices associated to each position of the sorting, as if applying it we'd get the ordered list
the second argsort translates the indices into ranks and the position goes back to the initial matrix's position.
e.g.
X = np.array([[30, 10, 20], 
              [5,   1,  9]])
argsort:
[[1 2 0]
 [1 0 2]]
argsort().argsort():
[[2 0 1]
 [1 0 2]]
"""
def spearman(x, y):
    xr = x.argsort().argsort().astype(float)
    yr = y.argsort().argsort().astype(float)
    rho = np.corrcoef(xr, yr)[0, 1]
    return rho
# EOF


"""
get_lagplots
From a correlation matrix, it returns the lagplot by averaging over the diagonals.
INPUT:
    - corr_mat: np.ndarray(float) -> (tpts x tpts) the auto-correlation or cross-correlation matrix
    - max_lag: int -> the maximum offset in tpts
    - min_datapts: int -> the minimum amount of points that a diagonal should have to be considered acceptable
    - symmetric: bool -> if we are computing a cross-correlation, corr_mat is not symmetric, otherwise it is
OUTPUT:
    - lagplot: np.ndarray -> (max_lag*2 + 1) if not symmetric, otherwise (max_lag +1), it's the correlation coefficient as a function of the lag
"""
def get_lagplot(corr_mat, max_lag=20, min_datapts=10, symmetric=False, summary_stat='mean'):
    # first sanity checks    
    get_lagplot_checks(corr_mat, max_lag, min_datapts)
    stat = choose_summary_stat(summary_stat)
    if not symmetric:
        d = np.diag(corr_mat)
        lagplot = np.zeros(max_lag*2 +1)
        lagplot[max_lag] = stat(d)
        for tau in range(1,max_lag+1): # +1 otherwise it selects the 0th diagonal
            d = np.diag(corr_mat, -tau)
            lagplot[max_lag+tau] = stat(d) # append because the lower triangular correspond to a positive offset between data1 and data2
            d = np.diag(corr_mat, tau)
            lagplot[max_lag-tau] = stat(d) # appendleft because the upper triangular correspond to a negative offset between data1 and data2
    else:
        d = np.diag(corr_mat)
        lagplot = np.zeros(max_lag +1)
        lagplot[0] = stat(d)
        for tau in range(1,max_lag+1): # +1 otherwise it selects the 0th diagonal
            d = np.diag(corr_mat, tau)
            lagplot[tau] = stat(d)
    return lagplot
# EOF


"""
get_lagplot_subset
Computes the lagplot (average along the diagonals) only within a specific window of the neural and model data.
INPUT:
    - corr_mat: np.ndarray(float) -> (tpts x tpts) the auto-correlation or cross-correlation matrix
    - neural_idx: list/array/range -> time points to consider along neural axis
    - model_idx: list/array/range -> time points to consider along model axis
    - max_lag: int -> the maximum offset in tpts
    - min_datapts: int -> the minimum amount of points that a diagonal should have to be considered acceptable
    - symmetric: bool -> if we are computing a cross-correlation, corr_mat is not symmetric, otherwise it is
OUTPUT:
    - lagplot: np.ndarray -> (max_lag*2 + 1) if not symmetric, otherwise (max_lag +1), it's the correlation coefficient as a function of the lag
"""
def get_lagplot_subset(corr_mat, neural_idx, model_idx=None, max_lag=20, min_datapts=10, summary_stat='mean'):
    nan_check(corr_mat)
    corr_mat_h, corr_mat_w = corr_mat.shape # rows are neural timepts, cols are model timepts
    neural_idx = np.array(neural_idx)
    if model_idx is None: # if no model idx is specified, choose all the models
        model_idx  = np.array(range(corr_mat_w))
    else:
        model_idx  = np.array(model_idx)
    # end if model_idx is None:
    
    lagplot = np.zeros(max_lag*2 + 1) # lagplot from -max_lag to +max_lag with the zero in the middle
    center = max_lag
    stat = choose_summary_stat(summary_stat)
    
    # compute each lag
    for tau in range(-max_lag, max_lag+1):
        diag_vals = []
        # build all possible aligned pairs
        for i_neu in neural_idx:
            i_mod = i_neu + tau  # model index candidate 
            if i_mod in model_idx:
                if 0 <= i_neu < corr_mat_h and 0 <= i_mod < corr_mat_w: # check bounds bc i_mod might be out of the matrix
                    diag_vals.append(corr_mat[i_neu, i_mod])
                # end if 0 <= i_neu < corr_mat_h and 0 <= i_mod < corr_mat_w:
            # end if i_mod in model_idx:
        # end for i_neu in neural_idx:

        if not diag_vals: # if there's nothing in there (because it didn't enter the 2nd if)
            raise IndexError("The maximum lag is larger than the selected matrix itself")
        elif len(diag_vals) < min_datapts:
            print_wise(f"The number of datapoints used to compute extreme offsets is < than {min_datapts}")
        # end if n_timepts < max_lag:
        lagplot[center - tau] = stat(diag_vals)
    # end for tau in range(-max_lag, max_lag+1):
    return lagplot



def split_integer(total: int, n: int):
    """Split total into n nearly equal integer parts."""
    base = total // n
    remainder = total % n
    # distribute the remainder (one extra for the first 'remainder' chunks)
    parts = [base + 1 if i < remainder else base for i in range(n)]
    return parts
# EOF

def make_intervals(total: int, n: int):
    parts = split_integer(total, n)
    intervals = []
    start = 0
    for p in parts:
        intervals.append((start, p))
        start = start+p
    return intervals



"""
get_timestamps
Constructs the timestamps for a time-series 
"""
def get_timestamps(n_timepts, sampling_rate):
    timestamps = np.arange(n_timepts)/sampling_rate
    return timestamps
#EOF


"""
get_upsampling_indices
Upsamples a signals by means of nearest neighbour timept.
"""
def get_upsampling_indices(n_old_timepts, old_rate, new_rate):
    old_timestamps = get_timestamps(n_old_timepts, old_rate)
    upsample_factor = new_rate/old_rate
    n_new_timepts = int(np.round(n_old_timepts * upsample_factor))
    new_timestamps = get_timestamps(n_new_timepts, new_rate)
    tree = cKDTree(old_timestamps[:, None])
    _, indices = tree.query(new_timestamps[:, None], k=1) # Query nearest old sample for each new time (like dsearchn)
    return indices
# EOF


def delete_empty_keys(data_dict):
    new_dict = {k: v for k, v in data_dict.items() if v.shape != (0,)}
    return new_dict


def binary_classification(x, y, n_splits, classification_function, *args, **kwargs):
    accuracy_list = []
    kf = KFold(n_splits=n_splits, shuffle=True) 
    for train_index, test_index in kf.split(x):
    # Split into training and testing sets
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # Initialize the model
        model = classification_function(*args, **kwargs)
    # Train the model
        model.fit(x_train, y_train)
    # Make predictions
        y_pred = model.predict(x_test)
    # Evaluate
        accuracy_list.append(accuracy_score(y_test, y_pred))
    avg_accuracy = np.mean(accuracy_list)
    return avg_accuracy


def binary_classification_over_time(condition_1, condition_2, channel_range, n_splits, classification_function, *args, **kwargs):
    min_trials = min(condition_1.shape[2], condition_2.shape[2]) # even the trials
#    condition_1, condition_2 = condition_1[:,:,:min_trials], condition_2[:,:,:min_trials]
    idx1 = np.random.choice(np.arange(0, condition_1.shape[2]), size=min_trials, replace=False)
    idx2 = np.random.choice(np.arange(0, condition_2.shape[2]), size=min_trials, replace=False)
    condition_1, condition_2 = condition_1[:,:,idx1], condition_2[:,:,idx2]
    if condition_1.shape[2] != condition_2.shape[2]:
        raise IndexError("The number of datapoints across conditions is different")
    condition_1_label = np.ones(condition_1.shape[2])
    condition_2_label = np.zeros(condition_2.shape[2])
    y = np.concatenate((condition_1_label, condition_2_label))
    x_timeseries = np.concatenate((condition_1, condition_2),axis=2)
    accuracy_over_time = []
    for i in range(x_timeseries.shape[1]):
        x = x_timeseries[channel_range[0]:channel_range[1], i, :].T
        avg_accuracy = binary_classification(x, y, n_splits, classification_function, *args, **kwargs)
        accuracy_over_time.append(avg_accuracy)
    accuracy_over_time = np.array(accuracy_over_time)
    return accuracy_over_time


"""
multivariate_ou
Generates a multivariate Ornstein–Uhlenbeck (OU) process with independent
dimensions, each following an exponentially correlated stochastic trajectory.

The OU process is defined by:
    x[t] = A * x[t-1] + noise
where A = exp(-dt / corr_length) controls the decay of temporal correlations.

INPUT:
    - T: float -> Total duration of the process (in arbitrary time units).
    - dim: int -> Number of independent OU dimensions to generate.
    - dt: float -> Time step used to discretize the process.
    - corr_length: float -> Correlation length (τ). Larger τ → slower decay → more temporal autocorrelation.
    - sigma: float (default 1.0) -> Noise scale governing the variance of the stochastic term.

OUTPUT:
    - x: np.ndarray (N, dim) -> Multivariate OU process, where N = int(T / dt) is the number of timepoints.
          Each column corresponds to one OU dimension.
"""
def multivariate_ou(T, dim, dt, corr_length, sigma=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    N = int(T / dt)
    x = np.zeros((N, dim))

    alpha = dt / corr_length      # decay factor
    A = np.exp(-alpha)            # autocorrelation coefficient

    for t in range(1, N):
        noise = sigma * np.sqrt(1 - A**2) * rng.standard_normal(dim)
        x[t] = A * x[t-1] + noise

    return x
# EOF


"""
Returns True if the list or np.array is empty, False otherwise.
Works for both lists and NumPy arrays.
"""
def is_empty(x):
    if x is None:
        return True
    try:
        # Works for np.array
        return x.size == 0
    except AttributeError:
        # If no size attribute, fallback to len() (lists, tuples)
        return len(x) == 0
# EOF




def get_device(verbose=False):
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if verbose:
        print_wise(f"device being used: {device}")
    return device
# EOF


""" 
load_npz_as_dict
Loads the saved npz file as a dict 
"""
def load_npz_as_dict(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
# EOF 

"""
decode_matlab_strings
Decodes MATLAB strings stored in a v7.3 .mat file (HDF5 format) into Python strings.
1) Iterates over HDF5 object references pointing to MATLAB char arrays
2) Reads the corresponding uint16 character codes
3) Converts character codes to Python characters and joins them into strings

INPUT:
- h5file: h5py.File -> open HDF5 file corresponding to a MATLAB v7.3 .mat file
- ref_array: np.ndarray -> array of HDF5 object references to MATLAB char arrays

OUTPUT:
- strings: list of str -> decoded MATLAB strings
"""
def decode_matlab_strings(h5file, ref_array):
    strings = []
    for ref in ref_array.squeeze():
        chars = h5file[ref][:]
        s = ''.join(chr(c) for c in chars.flatten()) # MATLAB chars are usually stored as Nx1 uint16
        strings.append(s)
    return strings


"""
index_gram
Indexes the upper-triangular elements of a Gram matrix in the same way pdist would do.
INPUT:
    - gram: np.ndarray(n, n) -> the Gram matrix
OUTPUT:
    - gram: np.ndarray(n(n-1)/2) -> vectorized upper-triangular gram 
"""
@njit
def index_gram(gram):
    n = gram.shape[0]
    n_pairs = n * (n - 1) // 2
    gram_vec = np.empty(n_pairs)
    counter = 0
    for i in range(n):
        for j in range(i+1, n):
            gram_vec[counter] = gram[i, j]
            counter += 1
    return gram_vec
# EOF


"""
cosine_sim
Computes the cosine dissimilarity (1-cosine_sim).
First it normalizes all the columns of x (axis=0), then it computes the dot-product gram.
It uses the fact that cos(theta) = dot(u,v)/(|u||v|) .
INPUT:
    - x: np.ndarray (d, n) -> data matrix of shape (features, points)
OUTPUT:
    - gram: np.ndarray(n(n-1)/2) -> vectorized upper-triangular gram with cosine similarity as distance metric
"""
@njit
def cosine_sim(x):
    norm = np.sqrt(np.sum(x**2, axis=0))
    x = x/norm
    gram = x.T @ x 
    gram = 1- gram
    gram = index_gram(gram)
    return gram
# EOF


"""
magnitude_diff

Computes the pairwise absolute difference of the L2 norms of the
column vectors of x and returns the upper-triangular entries
(excluding the diagonal).
INPUT:
    - x: np.ndarray (d, n) -> Data matrix
OUTPUT:
    - gram: np.ndarray, shape (n*(n-1)//2,) -> Vector containing the absolute difference of the norms | ||x_i|| - ||x_j|| |
"""
@njit
def magnitude_diff(x: np.ndarray) -> np.ndarray:
    d, n = x.shape
    norms = np.zeros(n)
    # Compute column-wise L2 norms
    for j in range(n): 
        s = 0.0
        for i in range(d): # loops through the dimensions of the current column j and sums their square until it finishes the dot prod x[:, j] @ x[:, j] 
            s += x[i, j] * x[i, j]
        # end for i in range(d):
        norms[j] = np.sqrt(s) 
    # end for j in range(n): 
    # Compute upper-triangular absolute differences
    n_pairs = n * (n - 1) // 2
    gram = np.zeros(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            gram[idx] = abs(norms[i] - norms[j])
            idx += 1
        # end for j in range(i+1, n):
    # end for i in range(n):
    return gram
# EOF


"""
compute_samples_sizes
Computes a sequence of sample sizes used for iterative or scaling analyses.

Generates a 1D array of sample sizes starting from `step_samples` up to
`max_size` (inclusive), with a fixed step.

INPUT:
- cfg: Cfg -> configuration object with required attributes:
    * step_samples: int
    * max_size: int

OUTPUT:
- n_samples: np.ndarray -> array of sample sizes
"""
def compute_samples_sizes(cfg):
    check_attributes(cfg, "step_samples", "max_size")
    n_samples = np.arange(cfg.step_samples, cfg.max_size +1, cfg.step_samples)
    return n_samples
# EOF


"""
get_module_by_path
Resolve a dotted module path with numeric indices.
Pass attribute path divided by dots and each digit is used index.
INPUT:
    - obj: object -> the object (any Python obj) from which we want to extract the attribute.
    - attribute_path: str -> the path from which we want to extract the attribute, separated by dots
        e.g. attribute_path = "layer.0.mlp.down_proj" -> model.layer[0].mlp.down_proj
OUTPUT:
    - obj: object -> the module extracted from the input object
"""
def get_module_by_path(obj, attribute_path: str):
    # it reassignes obj recursively as it goes deeper in the attributes
    for part in attribute_path.split("."): 
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
        # end 
    # end for part in path.split("."):
    return obj
# EOF


"""
get_triu_perms
Generate all unique unordered pairs (i, j) from a list, with i preceding j.
This corresponds to the lower-triangular (or upper-triangular) combinations
without self-pairs or duplicates.
INPUT:
    - in_list: list -> List of elements.
OUTPUT:
    - tuples_list: list[tuple] -> List of pairs (i, j) with i != j and each pair appearing once.
"""
def get_triu_perms(in_list: list):
    tuples_list = []
    for idx, i in enumerate(in_list):
        for j in in_list[:idx]:
            tuples_list.append((i,j))
        # end for j in in_list[:idx]:
    # end for idx, i in enumerate(in_list):
    return tuples_list
# EOF


# ---- HELPER FUNCTIONS ----
"""
bin_signal
(same as get_bins in neural_utils.preprocessing)
Computes time bins used to downsample (smooth) a time series to a new sampling frequency.
1) Computes the averaging window length based on the ratio between original and target fs
2) Generates bin edges spanning the full trial duration
3) Ensures the last bin reaches the end of the trial

INPUT:
- time_series: time_series -> time_series object containing the signal and sampling rate
- new_fs: float -> target sampling frequency (Hz)

OUTPUT:
- bins: np.ndarray -> array of integer indices defining bin edges along the time axis
"""
def bin_signal(time_series, new_fs):
    len_avg_window = time_series.fs /new_fs
    trial_duration = len(time_series)
    bins = np.round(np.arange(0, trial_duration, len_avg_window)).astype(int)  # bins the target trial with the required resolution, convert to int for later indexing
    if bins[-1] != trial_duration:
        bins = np.append(bins, int(trial_duration))  # adds the last time bin
    return bins
# EOF


"""
smooth_signal
(same as get_firing_rate in neural_utils.preprocessing)
Applies temporal smoothing to a time series by averaging samples within predefined bins.
1) Iterates over consecutive bin intervals
2) Extracts the corresponding time slices from the signal
3) Computes the mean activity within each bin
4) Stacks the averaged chunks along the time dimension

INPUT:
- time_series: time_series -> time_series object containing the signal array
- bins: np.ndarray -> array of integer bin edges defining smoothing windows

OUTPUT:
- smoothed_signal: np.ndarray -> time-smoothed signal with reduced temporal resolution
"""
def smooth_signal(time_series, bins):
    smoothed_signal = []
    
    for idx_bin, bin_start in enumerate(bins[:-1]): # the last el in bin is just the end of the trial, that's why the [:-1] indexing
        bin_end = bins[idx_bin + 1]
        curr_chunk = time_series.array[:,bin_start:bin_end, ...]  # slices the current chunk
        curr_avg_chunk = np.mean(curr_chunk, axis=1)  # computes the mean firing rate over the chunk
        smoothed_signal.append(curr_avg_chunk)    
    # end for idx, bin_start in enumerate(bins[:-1]):
    smoothed_signal = np.stack(smoothed_signal, axis=1) # stacks time in the columns
    return smoothed_signal
# EOF


"""
subsample_RDM
Starting from a full RDM, it extracts a smaller RDM using a subset of indices.
1) Selects the rows corresponding to the provided indices
2) Selects the matching columns (same indices)
3) Preserves the pairwise structure of the original RDM

INPUT:
- RDM: np.ndarray (N, N) -> full square representational dissimilarity matrix
- indices: array-like (K,) -> indices of conditions/items to keep

OUTPUT:
- subsampled_RDM: np.ndarray (K, K) -> sub-RDM restricted to the selected indices
"""
def subsample_RDM(RDM, indices):
    subsampled_RDM = RDM[np.ix_(indices, indices)]
    return subsampled_RDM
# EOF


def check_attributes(obj, *attrs):
    missing = [
        attr for attr in attrs
        if not hasattr(obj, attr) or getattr(obj, attr) is None
    ]
    if missing:
        raise AttributeError(
            f"{obj.__class__.__name__} has unset attributes: {missing}"
        )

def double_centering(G: np.ndarray, epsilon=10e-4) -> np.ndarray:
    N = G.shape[0]
    G_dcnt = (
        -0.5 * (np.eye(N) - 1 / N * np.ones(N)).T @ G @ (np.eye(N) - 1 / N * np.ones(N))
    )
    # to check if it's really centered
    control_double_centering(G_dcnt, epsilon)
    return G_dcnt
# EOF


"""
control_double_centering
Controls if G_dcnt is correctly double-centered up to a certain threshold epsilon.
INPUT:
- G_dcnt: np.ndarray -> double-centered Gram matrix
- epsilon: float -> threshold of acceptance

OUTPUT:
none 

"""
def control_double_centering(G_dcnt: np.ndarray, epsilon: float):
    if any(np.abs(np.sum(G_dcnt, axis=0)) > epsilon) or any(
        np.abs(np.sum(G_dcnt, axis=1)) > epsilon
    ):
        raise ValueError("the matrix isn't double-centered")
# EOF



# ---- CLASSES ----

"""
BrainAreas
Utility class for slicing neural data into predefined brain areas.
1) Loads brain-area channel indices from a YAML configuration file
2) Validates input rasters against the expected number of channels
3) Extracts and concatenates channel ranges corresponding to a given brain area

INPUT:
- monkey_name: str -> identifier used to select the correct brain-area mapping

OUTPUT (slice_brain_area):
- brain_area_response: np.ndarray -> subset of rasters corresponding to the selected brain area
"""
class BrainAreas:
    def __init__(self, monkey_name: str):
        self.monkey_name = monkey_name
        with open("../../brain_areas.yaml", "r") as f:
            config = yaml.safe_load(f)
        try:
            self.areas_idx = config[self.monkey_name]
            self.brain_areas = [k for k in self.areas_idx.keys() if k!='n_chan']
        except KeyError:
            raise KeyError(f"Monkey '{self.monkey_name}' not found.", f"Supported monkeys {list(config.keys())}") from None
        # end try:
    # EOF
    # --- GETTERS ---
    def get_brain_areas_idx(self):
        return self.areas_idx
    #EOF
    def get_brain_areas(self):
        return self.brain_areas
    #EOF
    # --- OTHER FUNCTIONS ---
    def slice_brain_area(self, rasters: "TimeSeries", brain_area_name: str):
        if rasters.get_array().shape[0] < self.areas_idx["n_chan"][0]:
            raise ValueError(f"Rasters of shape {rasters.get_array().shape} doesn't match the original number of channels ({self.areas_idx["n_chan"]}).")
        # end if rasters.shape[0] < self.areas_idx["n_chan"][0]:
        try:
            target_brain_area = self.areas_idx[brain_area_name]
        except KeyError:
            raise KeyError(f"Brain area '{brain_area_name}' not found for monkey '{self.monkey_name}'.", f"Supported brain areas: {list(self.areas_idx.keys())}") from None
            
        except TypeError:
            if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
                for idx in brain_area_name:
                    if idx > self.areas_idx["n_chan"][0]:
                        raise ValueError(f"Indices passed {brain_area_name} don't match the original number of channels ({self.areas_idx["n_chan"]}).")
                    # end if idx > self.areas_idx["n_chan"][0]:
                # end for idx in brain_area_name:
                target_brain_area = [brain_area_name] # it's setting the limits in terms of channels idx where we don't have precise info about a brain area, wrapping them in a list of lists
            else:
                raise TypeError(f"brain_area_name should be either a str or a list of len 2.")
            # end if isinstance(brain_area_name, list) and len(brain_area_name) == 2:
        # end try:
        brain_area_response = []
        for lims in target_brain_area:
            start, end = lims
            brain_area_response.append(rasters.get_array()[start:end, ...])
        # end for lims in target_brain_area:
        brain_area_response = np.concatenate(brain_area_response)
        brain_area_response = TimeSeries(brain_area_response, rasters.fs)
        return brain_area_response
    # EOF
# EOC

"""
TimeSeries
Container for multivariate neural time series data with flexible internal
representation (NumPy array or list of arrays).

Supports iteration over time points, temporal resampling, averaging across
dimensions, and autocorrelation-based analyses.

INTERNAL REPRESENTATION:
- type = "np"   : array shape (neurons, time, trials, ...)
- type = "list" : list of arrays, each array shape (neurons, trials, ...)

INPUT:
- array : np.ndarray (features, timepts [, trials]) | list[np.ndarray(feats [, trials])] 
    Neural data. Either a NumPy array with explicit time axis or a list of
    per-time-point arrays.
- fs : float
    Sampling frequency in Hz.

ATTRIBUTES:
- array : np.ndarray | list
    Stored data.
- fs : float
    Sampling frequency.
- type : str
    Data backend ("np" or "list").

NOTES:
- Many operations require NumPy-backed data and will raise if type == "list".
- Use `to_numpy()` to convert list-backed data to NumPy.
"""
class TimeSeries:
    def __init__(self, array, fs: float):
        if not isinstance(array, (np.ndarray, list)):
            raise TypeError(f"Unsupported type {type(array)}")
        elif isinstance(array, np.ndarray):
            self.type = "np"
        elif isinstance(array, list):
            self.type = "list"
        # end if not isinstance(array, (np.ndarray, list)):
        self.array = array
        self.fs = fs
    # EOF
    def type_check(self):
        if self.type != 'np':
            raise TypeError(f"{self.type} type is not supported for this operation")
        # end if self.type == 'np':
    # --- GETTERS ---
    def get_fs(self):
        return self.fs
    # EOF
    def get_duration_ms(self):
        return len(self) * 1000/self.fs 
    # EOF
    def get_duration_s(self):
        return len(self) / self.fs
    # EOF
    def get_array(self):
        return self.array
    # EOF

    # --- SETTERS ---
    def set_fs(self, new_fs):
        self.fs = new_fs
    # EOF
    def set_array(self, array):
        self.type_check()
        self.array = array
    # EOF

    # --- OTHER METHODS ---
    def __len__(self):
        """Number of time points."""
        if self.type == "np":
            try:
                return self.array.shape[1]
            except IndexError:
                return self.array.shape[0] # for the arrays that have just 1 feat, there is no possibility otherwise
        elif self.type == "list":
            return len(self.array)
    # EOF
    def __iter__(self):
        """Iterate over time points, yielding (neurons, trials)."""
        if self.type == "np":
            for t in range(len(self)):
                yield self.array[:, t, ...]
        elif self.type == "list":
            for t in self.array:
                yield t
        # end if self.type == "np":
    # EOF
    def __getitem__(self, t):
        """Return data at time index t: (neurons, trials)."""
        if self.type == "np":
            return self.array[:, t, ...]
        elif self.type =="list":
            return self.array[t]
        # end if self.type == "np":
    # EOF
    def to_numpy(self):
        if self.type == 'list':
            self.array = np.stack(self.array, axis=1)
            self.type = 'np'
        # end if self.type == 'list':
    # EOF
    def shape(self):
        if self.type == 'np':
            return self.get_array().shape
        else:
            raise TypeError("This TimeSeries is in the List format, before asking for the shape, convert it to_numpy()")
    # EOF
    def trial_avg(self):
        self.type_check()
        try:
            trial_avg = reduce(self.array, 'neurons time trials -> neurons time', 'mean')
            return trial_avg
        except EinopsError:
            raise EinopsError(f"Array of size {self.array.shape} doesn't have the trial dimension (2nd dimension).") from None            
        # end try:
        # EOF
    def neurons_avg(self):
        self.type_check()
        neurons_avg = reduce(self.get_array(), 'neurons time ... -> 1 time ...', 'mean')
        return neurons_avg    
    # EOF
    def overall_avg(self):
        self.type_check()
        overall_avg = reduce(self.get_array(), 'neurons time ... -> time', 'mean')
        return overall_avg
    # EOF
    def z_score_feats(self):
        X_array = self.get_array()  # shape: (feats, time) or (feats, time, trials)
        # compute mean and std across all non-feature axes
        axes = tuple(range(1, X_array.ndim))  # all axes except feats
        m = X_array.mean(axis=axes, keepdims=True)   # shape compatible for broadcasting
        std = X_array.std(axis=axes, keepdims=True)
        X_z = (X_array - m) / std
        self.set_array(X_z)
        return self.get_array()
    # EOF
    def resample(self, new_fs):
        self.type_check()
        if new_fs < self.fs: # smoothing
            bins = bin_signal(self, new_fs)
            new_array = smooth_signal(self, bins)
        elif new_fs > self.fs: # upsampling
            upsampling_indices = get_upsampling_indices(len(self), self.fs, new_fs) # ADD modify to make if become a list
            new_array = self.array[:,upsampling_indices, ...]
        else:
            new_array = self.array
        # end if new_fs < self.fs:
        # in-place modifications
        self.set_array(new_array) 
        self.set_fs(new_fs) # updates the fs
    # EOF
    def autocorr(self, metric: str ='correlation', max_lag: int = 20):
        self.to_numpy()
        ac_mat = autocorr_mat(self.array, metric=metric) 
        ac_line = get_lagplot(ac_mat, max_lag=max_lag, symmetric=True)
        return ac_mat, ac_line
    # EOF
    def lagged_corr(self, other: "TimeSeries", metric: str = 'correlation'):
        self.to_numpy(); other.to_numpy()
        ac_mat = autocorr_mat(self.array, data2=other.array, metric=metric)
        return ac_mat 
    # EOF
    """
    delay_embeddings
    Construct delay-embedded version of the time series by stacking temporal windows
    around each timepoint.

    INPUT:
        - lags: tuple (lag_start, lag_end) -> Temporal window relative to each timepoint.
            Example:
                (-2, 0)   -> uses [t-2, t-1, t]
                (-2, 2)   -> uses [t-2, ..., t+2]

        - pad_mode: str | None (default=None) -> Padding strategy at the boundaries.
            Determines how values outside the signal are handled.

            Cases:
                - None:
                    No padding. Only valid timepoints are considered.
                    Output duration is reduced.
                    Example:
                        lags = (-2, 0), T = 100 -> output length = 98

                - "edge":
                    Repeats boundary values.
                    Example:
                        x = [1, 2, 3], lags = (-1, 0)
                        padded -> [1, 1, 2, 3]

                - "reflect":
                    Mirrors the signal at the boundaries.
                    Example:
                        x = [1, 2, 3], lags = (-2, 0)
                        padded -> [3, 2, 1, 2, 3]

                - "constant":
                    Pads with zeros (or constant value).
                    Example:
                        x = [1, 2, 3], lags = (-2, 0)
                        padded -> [0, 0, 1, 2, 3]

    OUTPUT:
        - TimeSeries -> Delay-embedded time series of shape
            (features * window_size, timepoints_new),
            where window_size = lag_end - lag_start + 1.
            Sampling frequency is preserved.
    """
    def delay_embeddings(self, lags, pad_mode=None):
        X = self.get_array()
        if X.ndim == 3:
            raise ValueError("Not implemented for trials yet")
        T = len(self)
        lag_start, lag_end = lags
        if pad_mode is None:
            X_padded = X
            time_indices = range(-lag_start, T - lag_end) # given the lags, the edges won't be considered
            pad_left = 0 # useful to compute the center later
        else:
            pad_left  = -lag_start
            pad_right = lag_end
            X_padded = np.pad(
                X,
                ((0, 0), (pad_left, pad_right)),
                mode=pad_mode
            )
            time_indices = range(T) # considers all the timepoints
        # end if pad:
        ts_delay_emb = []
        # loops through all the available timepoints and embeds them in their delays
        for t in time_indices:
            center = t + pad_left
            window = X_padded[:, center + lag_start : center + lag_end + 1]
            ts_delay_emb.append(window.flatten()) # takes the features over the points -lags(t):lags(t) and flattens their features
        # end for t in time_indices:
        Z = np.stack(ts_delay_emb, axis=1)
        return TimeSeries(Z, self.get_fs())
    # EOF
# EOC

"""Small function for checking if two TimeSeries have the same length and frequency"""
def compatible_TimeSeries_check(X: TimeSeries, Y: TimeSeries):
    if len(X) != len(Y):
        raise ValueError(f"X and Y TimeSeries must have the same length, got {len(X)} and {len(Y)}. ")
    if X.get_fs() != Y.get_fs():
        raise ValueError(f"X and Y TimeSeries must have the same sampling frequency, got X_fs={X.get_fs()} and Y_fs={Y.get_fs()}.")
# EOF 

"""
get_lags
Returns the range of temporal lags used for dynamic modeling.

INPUT:
    - max_lag: int -> maximum temporal lag considered
    - symmetric: bool -> if True returns only positive lags (0 → max_lag), 
                         otherwise returns symmetric lags (-max_lag → max_lag)

OUTPUT:
    - lags_range: range(int) -> iterable containing the lag values
"""
def get_lags(max_lag, symmetric=False):
    if symmetric:
        lags_range = range(max_lag+1)
    else:
        lags_range = range(-max_lag, max_lag+1)
    # end if symmetric:
    return lags_range
# EOF


"""
shift_concatenate_xy
Shifts temporally all the trials of two time-series x (model), y (neural signal) according to a lag tau. 
INPUT:
    - x: TimeSeries (features, timepoints, trials) -> regressor (i.e. model) time series 
    - y: TimeSeries (features, timepoints, trials) -> signal (i.e. neurons) time series
    - tau: int -> temporal lag
        tau < 0 we are using x future to predict current y
        tau = 0 we are using x present to predict current y
        tau > 0 we are using x past to predict current y
OUTPUT:
    - x_shifted: np.ndarray ((overlapping timepoints * trials), features) -> regressor (i.e. model) time series. 
                            overlapping timepoints = the timepoints that still have a correspondent in the other time series
                            the different trials are vectorized in the 0th dimension in the format regression_obj wants
    - y_shifted: np.ndarray ((overlapping timepoints * trials), features) -> signal (i.e. neurons) time series
"""
def shift_concatenate_xy(x, y, tau, transpose=True):
    x_shifted_tot = []
    y_shifted_tot = []
    
    for i_trial in range(x.get_array().shape[2]):
        curr_x = x.get_array()[:,:,i_trial]
        curr_y = y.get_array()[:,:,i_trial]
        x_shifted, y_shifted = shift_xy(curr_x, curr_y, tau)
        if transpose:
            x_shifted = np.ascontiguousarray(x_shifted.T)
            y_shifted = np.ascontiguousarray(y_shifted.T)
        # end if transpose:
        x_shifted_tot.append(x_shifted)
        y_shifted_tot.append(y_shifted)
    # end for i in range(data.shape[2]):
    axis = 0 if transpose else 1 # concatenate along the rows if transpose, otherwise the columns -> result: 1st case) NxD; 2nd case) DxN
    x_shifted_tot = np.concatenate(x_shifted_tot, axis=axis)
    y_shifted_tot = np.concatenate(y_shifted_tot, axis=axis)
    return x_shifted_tot, y_shifted_tot
# EOF


"""
shift_xy
Shifts temporally two time-series x (model), y (neural signal) according to a lag tau. 
INPUT:
    - x: np.ndarray (features, timepoints) -> regressor (i.e. model) time series 
    - y: np.ndarray (features, timepoints) -> signal (i.e. neurons) time series
    - tau: int -> temporal lag
        tau < 0 we are using x future to predict current y
        tau = 0 we are using x present to predict current y
        tau > 0 we are using x past to predict current y
OUTPUT:
    - x_shifted: np.ndarray (features, overlapping timepoints) -> regressor (i.e. model) time series. 
                            overlapping timepoints = the timepoints that still have a correspondent in the other time series
    - y_shifted: np.ndarray (features, overlapping timepoints) -> signal (i.e. neurons) time series
"""
def shift_xy(x, y, tau):
    if tau > 0:  # if positive lag
        x_shifted = x[:,:-tau] # mod is shifted towards the right (so present neural is being compared with past model)
        y_shifted = y[:,tau:]   
    elif tau == 0:  # Handle L=0 case explicitly
        x_shifted = x
        y_shifted = y
    else:  # tau < 0
        x_shifted = x[:,-tau:] # mod is shifted towards the left (so present neural is being compared with future model)
        y_shifted = y[:,:tau]
    # if tau > 0:  # if positive lag
    return x_shifted, y_shifted
#EOF


def softmax(x, T=1.0):
    x = np.asarray(x)
    x = x / T
    e = np.exp(x - np.max(x))  # numerical stability
    return e / e.sum()
# EOF
