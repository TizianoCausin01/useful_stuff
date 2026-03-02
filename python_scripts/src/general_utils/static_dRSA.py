import os, yaml, sys
import numpy as np
from scipy.spatial.distance import squareform

sys.path.append("..")
from general_utils.utils import print_wise, TimeSeries, dRSA, load_img_natraster, check_attributes, subsample_RDM
from general_utils.II import dynInformationImbalance


"""
init_whole_neural_RDM
Initializes and computes the full neural RDM time series for a given brain area.

1) Creates a dRSA object using the RDM metric specified in the configuration
2) Computes the signal RDM at each time point of the input TimeSeries
3) Converts each vectorized RDM into its square (matrix) form

INPUT:
- area_rasters: TimeSeries -> neural activity restricted to a single brain area
- cfg: Cfg -> configuration object with required attributes:
    * RDM_metric: str

OUTPUT:
- drsa_obj: dRSA -> dRSA object containing the signal RDM time series
- whole_RDM_signal: list[np.ndarray] -> list of square-form RDMs (one per time point)
"""
def init_whole_neural_RDM(area_rasters, cfg):
    check_attributes(cfg, "RDM_metric")
    drsa_obj = dRSA(cfg.RDM_metric)
    drsa_obj.compute_RDM_timeseries(area_rasters, "signal")
    whole_RDM_signal = [squareform(RDM_t) for RDM_t in drsa_obj.get_RDM_timeseries("signal")]
    return drsa_obj, whole_RDM_signal
# EOF

"""
similarity_subsamples_loop
Computes static dRSA time courses for multiple random subsamples of trials.

For each requested subsample size, the function:
1) Randomly samples trials without replacement
2) Subsamples neural and model RDMs accordingly
3) Computes the static dRSA time series
4) Repeats the procedure for a fixed number of iterations

INPUT:
- n_samples: np.ndarray[int] -> array of subsample sizes
- iter_dict: dict[int, np.ndarray] -> dictionary mapping sample size to results array
- whole_RDM_signal: list[np.ndarray] -> list of square-form neural RDMs (one per time point)
- whole_RDM_model: np.ndarray -> square-form model RDM
- cfg: Cfg -> configuration object with required attributes:
    * n_iter: int
    * RDM_metric: str
    * new_fs: float

OUTPUT:
- iter_dict: dict[int, np.ndarray] -> updated dictionary where each key k maps to
  an array of shape (n_iter, n_timepoints) containing static dRSA values
"""
def similarity_subsamples_loop(n_samples: np.ndarray[int], iter_dict: dict[int, np.ndarray], whole_RDM_signal: list[np.ndarray], whole_RDM_model: np.ndarray, cfg):
    check_attributes(cfg, "n_iter", "RDM_metric", "new_fs")
    drsa_obj = dRSA(cfg.RDM_metric)
    n_trials = whole_RDM_signal[0].shape[0]
    n_tpts = len(whole_RDM_signal)
    for k in n_samples: 
        curr_ns = np.empty((cfg.n_iter, n_tpts))
        for i_iter in range(cfg.n_iter):
            curr_idx = np.random.choice(n_trials, size=k, replace=False)
            assert len(np.unique(curr_idx)) == k # sanity check that we have unique elements
            curr_neu = [squareform(subsample_RDM(RDM_t, curr_idx)) for RDM_t in whole_RDM_signal]
            curr_neu = TimeSeries(curr_neu, cfg.new_fs)
            drsa_obj.set_RDM_timeseries(curr_neu, "signal")
            curr_mod = squareform(subsample_RDM(whole_RDM_model, curr_idx))
            drsa_obj.set_RDM(curr_mod, "model")
            curr_ns[i_iter, :] = drsa_obj.compute_static_dRSA().array
        # end for i_iter in range(n_iter):
        iter_dict[k] = curr_ns
    # end for k in n_samples: 
    return iter_dict
# EOF


"""
similarity_subsamples_par
Parallel wrapper for subsampled static dRSA computation for a single model layer.

1) Loads model features and computes the full model RDM
2) Computes the static dRSA using all trials (reference condition)
3) Runs subsampled dRSA computations across multiple sample sizes
4) Saves the results to disk in compressed NPZ format

INPUT:
- paths
- rank: int -> process rank (used for controlled logging in parallel execution)
- layer_name: str -> name of the model layer being analyzed
- drsa_obj: dRSA -> initialized dRSA object
- whole_RDM_signal: list[np.ndarray] -> list of square-form neural RDMs (one per time point)
- n_samples: np.ndarray[int] -> array of subsample sizes
- cfg: Cfg -> configuration object with required attributes:
    * new_fs: float
    * monkey_name: str
    * model_name: str
    * img_size: int
    * n_iter: int
    * brain_area: str

OUTPUT:
- None
Side effects:
- Saves a compressed .npz file containing the subsampling results
"""
def similarity_subsamples_par(paths: dict[str: str], rank: int, layer_name: str, drsa_obj: dRSA, whole_RDM_signal: list[np.ndarray], n_samples: np.ndarray[int], cfg):
    check_attributes(cfg, "new_fs", "monkey_name", "model_name", "img_size", "n_iter", "brain_area", "RDM_metric")
    dict_savename = f"{paths['livingstone_lab']}/tiziano/results/subsampling_{cfg.new_fs}Hz_{n_samples[0]}-{n_samples[-1]}_{cfg.n_iter}iter_{cfg.monkey_name}_{cfg.date}_{cfg.brain_area}_{cfg.RDM_metric}_{cfg.model_name}_{cfg.img_size}_{layer_name}.npz"
    if os.path.exists(dict_savename):
        print_wise(f"model already exists at {dict_savename}", rank=rank)
    else:
        feats_filename = f"{paths['livingstone_lab']}/tiziano/models/{cfg.monkey_name}_{cfg.date}_{cfg.model_name}_{cfg.img_size}_{layer_name}_features_{cfg.pooling}pool.npz"
        feats = np.load(feats_filename)["arr_0"]
        drsa_obj.compute_RDM(feats, "model")
        whole_RDM_model = squareform(drsa_obj.get_RDM("model"))
        iter_dict = {ns: np.empty((0,)) for ns in n_samples}
        n_trials = feats.shape[1]
        iter_dict[n_trials] = drsa_obj.compute_static_dRSA().array
        print_wise(f"computed the whole static dRSA for layer {layer_name}", rank=rank)
        del drsa_obj
        iter_dict = similarity_subsamples_loop(n_samples, iter_dict, whole_RDM_signal, whole_RDM_model, cfg)
        np.savez_compressed(dict_savename, **{str(k): v for k, v in iter_dict.items()})
        print_wise(f"computed all iterations for layer {layer_name}, \nsaved at {dict_savename}", rank=rank)
    # end if os.path.exists(dict_savename):
# EOF


"""
init_static_dRSA_dynII
Initializer for static dRSA and dynamic Information Imbalance (dynII) objects.
This function sets up the shared signal-side computations required for both
static dRSA and dynamic II analyses. Specifically:
1) Initializes a dRSA object with the specified signal and model RDM metrics
2) Computes the signal RDM time series from the provided brain-area raster
3) Initializes a dynInformationImbalance object
4) Transfers the signal RDM time series to the dynII object
5) Computes the signal-side distance-rank time series for dynII

INPUT:
- ba_raster: TimeSeries
    Neural data raster for a given brain area
    Shape typically (n_trials, n_timepoints, n_features) or equivalent
- signal_RDM_metric: str or callable
    Distance metric used to compute signal RDMs
- model_RDM_metric: str or callable
    Distance metric used to compute model RDMs
- k: int
    Number of nearest neighbors used for Information Imbalance computation

OUTPUT:
- drsa_obj: dRSA
    Initialized dRSA object with signal RDM time series computed
- dyn_ii_obj: dynInformationImbalance
    Initialized dynII object with signal distance-rank time series computed

Side effects:
- Computes and stores signal-side RDMs and distance ranks inside the objects
"""
def init_static_dRSA_dynII(rank, ba_raster: "TimeSeries", signal_RDM_metric, model_RDM_metric, k: int) -> tuple["dRSA", "dynInformationImbalance"]:
    drsa_obj = dRSA(signal_RDM_metric, model_RDM_metric)
    dyn_ii_obj = dynInformationImbalance(signal_RDM_metric, model_RDM_metric, k)
    if rank != 0: # to avoid that the master computes it (memory saving) 
        drsa_obj.compute_RDM_timeseries(ba_raster, "signal")
        dyn_ii_obj.set_RDM_timeseries(drsa_obj.get_RDM_timeseries("signal"), "signal")
        dyn_ii_obj.compute_distance_ranks_timeseries("signal")
    return drsa_obj, dyn_ii_obj
# EOF

"""
compute_static_dRSA_dynII
Compute static dRSA and dynamic Information Imbalance for a single model layer.
This function performs the model-side computations required to compare a given
model layer to previously computed neural (signal) RDM time series.
1) Loads model features for the specified layer
2) Reorders features according to the provided trial index mapping
3) Computes the model RDM
4) Computes the static dRSA between signal and model RDMs
5) Sets the model RDM in the dynII object
6) Computes model-side distance ranks
7) Computes both directions of static dynamic Information Imbalance (A→B, B→A)

INPUT:
- paths: dict
- layer_name: str
    Name of the model layer being analyzed
- drsa_obj: dRSA
    Initialized dRSA object with signal RDM time series already computed
- dyn_ii_obj: dynInformationImbalance
    Initialized dynII object with signal distance-rank time series already computed
- idx_ord: list[int] or np.ndarray[int]
    Index mapping used to reorder model trials to match neural presentation order
- folder_name: str
    Name of the stimulus folder used to build the model feature filename
- model_name: str
    Name of the model (e.g., 'vit_l_16', 'resnet50')
- img_size: int
    Input image size used for the model
- pooling: str
    Pooling strategy used when extracting model features

OUTPUT:
- drsa: TimeSeries 
    Static dRSA time course
- dyn_ii_A2B: TimeSeries
    Static dynamic Information Imbalance from signal → model
- dyn_ii_B2A: TimeSeries 
    Static dynamic Information Imbalance from model → signal

Raises:
- AttributeError
    If required signal-side attributes are missing from drsa_obj or dyn_ii_obj
"""
def compute_static_dRSA_dynII(paths, rank, layer_name, drsa_obj, dyn_ii_obj, idx_ord, monkey_name, date, brain_area, folder_name, model_name, img_size, pooling) -> tuple["TimeSeries", "TimeSeries", "TimeSeries"]:
    save_name_A2B = f"{paths['livingstone_lab']}/tiziano/results/dynII_A2B_k{dyn_ii_obj.k}_{drsa_obj.signal_RDM_metric}-{drsa_obj.model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{dyn_ii_obj.get_RDM_timeseries("signal").get_fs()}Hz.npz"
    save_name_B2A = f"{paths['livingstone_lab']}/tiziano/results/dynII_B2A_k{dyn_ii_obj.k}_{drsa_obj.signal_RDM_metric}-{drsa_obj.model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{dyn_ii_obj.get_RDM_timeseries("signal").get_fs()}Hz.npz"
    save_name_drsa = f"{paths['livingstone_lab']}/tiziano/results/static_dRSA_{drsa_obj.signal_RDM_metric}-{drsa_obj.model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{drsa_obj.get_RDM_timeseries("signal").get_fs()}Hz.npz"
    if os.path.exists(save_name_drsa) and os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
        print_wise(f"model already exists at {save_name_A2B}", rank=rank)
    else:
        if not hasattr(drsa_obj, "signal_RDM_timeseries"):
            raise AttributeError("drsa_obj must have 'signal_RDM_timeseries'")
        # end if not hasattr(drsa_obj, "signal_RDM_timeseries"):
        if not hasattr(dyn_ii_obj, "signal_distance_ranks_timeseries"):
            raise AttributeError("dyn_ii_obj must have 'signal_distance_ranks_timeseries'")
        # end if not hasattr(dyn_ii_obj, "signal_distance_ranks_timeseries"):
        feats_filename = f"{paths['livingstone_lab']}/tiziano/models/{folder_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
        features = np.load(feats_filename)["arr_0"][:, idx_ord]
        drsa_obj.compute_RDM(features, "model")
        drsa = drsa_obj.compute_static_dRSA()
        dyn_ii_obj.set_RDM(drsa_obj.get_RDM("model"), "model")
        dyn_ii_obj.compute_distance_ranks("model")
        dyn_ii = dyn_ii_obj.compute_both_static_dynII()
        np.savez_compressed(save_name_drsa, drsa.get_array())
        np.savez_compressed(save_name_A2B, dyn_ii[0].get_array())
        np.savez_compressed(save_name_B2A, dyn_ii[1].get_array())
        print_wise(f"model saved at {save_name_A2B} and {save_name_drsa}", rank=rank)
    # end if os.path.exists(save_name_drsa) and os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
# EOF


def init_static_dRSA(ba_raster: "TimeSeries", signal_RDM_metric, model_RDM_metric) -> "dRSA":
    drsa_obj = dRSA(signal_RDM_metric, model_RDM_metric)
    drsa_obj.compute_RDM_timeseries(ba_raster, "signal")
    return  drsa_obj
# EOF

def compute_static_dRSA(paths: dict[str: str], rank: int, layer_name: str, drsa_obj, idx_ord: np.ndarray[int], monkey_name, date: str, brain_area: str, folder_name: str, model_name: str, img_size: int, pooling: str) -> "TimeSeries":    
    save_name = f"{paths['livingstone_lab']}/tiziano/results/static_dRSA_{drsa_obj.signal_RDM_metric}-{drsa_obj.model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{drsa_obj.get_RDM_timeseries('signal').get_fs()}Hz.npz"
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
    else:
        if not hasattr(drsa_obj, "signal_RDM_timeseries"):
            raise AttributeError("drsa_obj must have 'signal_RDM_timeseries'")
        # end if not hasattr(drsa_obj, "signal_RDM_timeseries"):
        feats_filename = f"{paths['livingstone_lab']}/tiziano/models/{folder_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
        features = np.load(feats_filename)["arr_0"][:, idx_ord]
        drsa_obj.compute_RDM(features, "model")
        drsa = drsa_obj.compute_static_dRSA()
        np.savez_compressed(save_name, drsa.get_array())
        print_wise(f"model saved at {save_name}", rank=rank)
    # end if os.path.exists(save_name):
# EOF




