import os, yaml, sys
import numpy as np

sys.path.append("..")
from general_utils.utils import print_wise, TimeSeries, create_RDM, spearman, check_attributes


"""
RSA
Implements Representational Similarity Analysis (RSA) between a signal RDM
and a model RDM.

Handles RDM construction, metric configuration, and similarity computation.

INPUT:
- signal_RDM_metric : str
    Distance/similarity metric used to compute the signal RDM.
- model_RDM_metric : str | None
    Metric for model RDM (defaults to signal_RDM_metric).
- RSA_metric : str
    Similarity metric between RDMs ("correlation" or "spearman").
- signal_RDM : np.ndarray | None
    Precomputed signal RDM (vector form).
- model_RDM : np.ndarray | None
    Precomputed model RDM (vector form).

ATTRIBUTES:
- signal_RDM_metric, model_RDM_metric : str
- RSA_metric : str
- signal_RDM, model_RDM : np.ndarray
- similarity : float (set after compute_RSA)

NOTES:
- RDMs are assumed to be in vector (condensed) form.
- Attribute existence is validated via `check_attributes`.
"""
class RSA:
    def __init__(
            self, 
            signal_RDM_metric: str, 
            model_RDM_metric: str=None, 
            RSA_metric: str='correlation', 
            signal_RDM: np.ndarray = None, 
            model_RDM: np.ndarray = None,
            ):
        self.signal_RDM_metric = signal_RDM_metric
        if model_RDM_metric is None:
            self.model_RDM_metric = signal_RDM_metric
        else:
            self.model_RDM_metric = model_RDM_metric
        # end if model_RDM_metric is None:
        self.RSA_metric = RSA_metric
        self.signal_RDM = signal_RDM
        self.model_RDM = model_RDM
    # EOF
    
    # --- GETTERS ---
    def get_RDM_metric(self, RDM_type):
        check_RDM_type(RDM_type)
        attr = f"{RDM_type}_RDM_metric"
        return getattr(self, attr)
    # EOF
    def get_RSA_metric(self):
        return self.RSA_metric
    # EOF
    def get_RDM(self, RDM_type: str):
        check_RDM_type(RDM_type)
        attr = f"{RDM_type}_RDM"
        return getattr(self, attr)
    # EOF

    # --- SETTERS ---
    def set_RDM_metrics(self, metric: str, RDM_type: str):
        check_RDM_type(RDM_type)
        attr = f"{RDM_type}_RDM_metric"
        setattr(self, attr, metric)
    # EOF
    def set_RDM(self, RDM: np.ndarray, RDM_type: str):
        check_RDM_type(RDM_type)
        if RDM_type == "signal":
            self.signal_RDM = RDM
        elif RDM_type == "model":
            self.model_RDM = RDM
        else:
            raise ValueError("Supported RDM types are 'signal' or 'model'")
        # end if RDM_type == "signal":
    # EOF
    
    # --- OTHER METHODS ---
    def compute_RDM(self, data: np.ndarray, RDM_type: str):
        check_RDM_type(RDM_type)
        check_attributes(self, f"{RDM_type}_RDM_metric")
        metric = getattr(self, f"{RDM_type}_RDM_metric")
        attr = f"{RDM_type}_RDM"
        RDM = create_RDM(data, metric)
        setattr(self, attr, RDM)
        return RDM
    # EOF
    def compute_both_RDMs(self, signal: np.ndarray, model: np.ndarray):
        check_attributes(self, "signal_RDM_metric", "model_RDM_metric")
        self.compute_RDM(signal, "signal")
        self.compute_RDM(model, "model")
    # EOF
    def compute_RSA(self): 
        check_attributes(self, "signal_RDM_metric", "model_RDM_metric", "RSA_metric", "signal_RDM", "model_RDM")
        if self.RSA_metric == 'correlation':
            similarity = np.corrcoef(self.signal_RDM, self.model_RDM)[0,1]
        elif self.RSA_metric == 'spearman':
            similarity = spearman(self.signal_RDM, self.model_RDM)
        else:
            raise TypeError(f"{self.RSA_metric} not supported.")
        # if self.RSA_metric == 'correlation':
        self.similarity = similarity
        return similarity
    # EOF
    def squareform(self, RDM_type: str):
        check_attributes(self, f"{RDM_type}_RDM")
        target_RDM = self.get_RDM(RDM_type)
        return squareform(target_RDM)
    # EOF
# EOC


"""
dRSA
Extends RSA to time-resolved Representational Similarity Analysis.

Computes RDMs at each time point and evaluates lagged or static RSA across time.

INPUT:
- signal_RDM_metric : str
- model_RDM_metric : str | None
- RSA_metric : str
- signal_RDM_timeseries : TimeSeries | None
    Time-resolved signal RDMs.
- model_RDM_timeseries : TimeSeries | None
    Time-resolved model RDMs.
- model_RDM_static : np.ndarray | None
    Static model RDM for time-resolved RSA.

ATTRIBUTES:
- signal_RDM_timeseries : TimeSeries
- model_RDM_timeseries : TimeSeries
- model_RDM_static : np.ndarray
- dRSA_mat : np.ndarray
- static_dRSA : TimeSeries

NOTES:
- TimeSeries objects are expected to iterate over vectorized RDMs.
- Correlation-based static dRSA is vectorized; Spearman is loop-based.
"""
class dRSA(RSA):
    def __init__(
            self, 
            signal_RDM_metric: str, 
            model_RDM_metric: str=None, 
            RSA_metric: str='correlation', 
            signal_RDM_timeseries: TimeSeries = None, 
            model_RDM_timeseries: TimeSeries = None,
            model_RDM_static: np.ndarray = None,
            ):
        super().__init__(signal_RDM_metric, model_RDM_metric, RSA_metric)
        self.signal_RDM_timeseries = signal_RDM_timeseries
        self.model_RDM_timeseries = model_RDM_timeseries
        self.model_RDM_static = model_RDM_static
    # EOF

    # --- GETTERS ---
    def get_RDM_timeseries(self, RDM_type: str):
        check_RDM_type(RDM_type)
        attr = f"{RDM_type}_RDM_timeseries"
        return getattr(self, attr)
    # EOF
    
    # --- SETTERS ---
    def set_RDM_timeseries(self, RDM_timeseries: TimeSeries, RDM_type: str):
        check_RDM_type(RDM_type)
        attr = f"{RDM_type}_RDM_timeseries"
        setattr(self, attr, RDM_timeseries)
    # EOF

    # --- OTHER FUNCTIONS
    def compute_RDM_timeseries(self, signal: TimeSeries, RDM_type: str):
        check_RDM_type(RDM_type)
        metric = getattr(self, f"{RDM_type}_RDM_metric")
        signal_fs = signal.get_fs()
        RDMs_list = []
        for t in range(signal.array.shape[1]): 
            RDM = create_RDM(
                np.ascontiguousarray(signal.array[:,t,:]), 
                metric
            )
            RDMs_list.append(RDM)
        RDMs_list = TimeSeries(RDMs_list, signal_fs)
        attr = f"{RDM_type}_RDM_timeseries"
        setattr(self, attr, RDMs_list)
    # EOF
    def compute_both_RDM_timeseries(self, signal: TimeSeries, model: TimeSeries):
        self.compute_RDM_timeseries(signal, "signal")
        self.compute_RDM_timeseries(model, "model")
    # EOF
    def compute_dRSA(self):
        check_attributes(self, "signal_RDM_timeseries", "model_RDM_timeseries")
        self.dRSA_mat = self.signal_RDM_timeseries.lagged_corr(self.model_RDM_timeseries, self.RSA_metric)
        return self.dRSA_mat
    # EOF
    def compute_static_dRSA(self):
        check_attributes(self, "signal_RDM_timeseries", "model_RDM")
        static_dRSA = []
        if self.RSA_metric == 'correlation': 
            for t in range(len(self.signal_RDM_timeseries)):
                static_dRSA.append(np.corrcoef(self.signal_RDM_timeseries.array[t], self.model_RDM)[0,1])
        elif self.RSA_metric == 'spearman':
            for t in self.signal_RDM_timeseries:
                static_dRSA.append(spearman(t, self.model_RDM))
            # end for i_time in signal.shape[1]:
        static_dRSA = np.array(static_dRSA)
        # end if metric == 'correlation':
        static_dRSA = TimeSeries(static_dRSA, self.signal_RDM_timeseries.fs)
        self.static_dRSA = static_dRSA
        return static_dRSA
    # EOF
# EOC


def check_RDM_type(RDM_type: str):
    if RDM_type not in ("signal", "model"):
        raise ValueError(f"{RDM_type} is not a supported RDM_type, you can choose either 'signal' or 'model'")
    # end if RDM_type != "signal" & RDM_type!="model":
# EOF
