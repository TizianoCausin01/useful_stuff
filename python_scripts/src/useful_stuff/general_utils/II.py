import os, yaml, sys
import numpy as np
from scipy.spatial.distance import squareform
sys.path.append("../..")
from useful_stuff.general_utils.utils import TimeSeries, print_wise
from useful_stuff.general_utils.RSA import RSA, dRSA

"""
InformationImbalance
Computes the Information Imbalance (II) between two representational spaces
according to Glielmo et al. 2022 and Del Tatto et al. 2024.
Computes asymmetric information imbalance based on distance ranks
derived from representational dissimilarity matrices (RDMs).
About the computation of distance ranks...

compute_distance_ranks
Creates a matrix of distance ranks from an RDM. 
I.e. the element (i,j) will tell you the rank of the point i 
with respect to the point j. So the ith column tells you the
distance rank of the points with respect to the ith point.
The matrix is not symmetric anymore and only the columns are 
interpretable. The diagonal ranks are filled with N+1 vals.
We take two argsort because 
- 1st argsort gives you the row-wise (i.e. each column is treated 
independently) index that if, applied to the original vector, 
would give you a vector with the values of the original vector 
in increasing order.  
So here in each column: the position in the vector gives you the
value rank, while its value the index in the original vector.
- 2nd argsort gives you the distance rank of the element i (row) in
the neighborhood of the element j (col).
In vector terms, it yields the vector that if applied to the output
of the first argsort, would give you np.arange(0, N) (the order of 
the ranks). So the returned vector has again the value rank in the
position and the index in its value, but since it's applied to the 
output of the first argsort, it returns the distance ranks.
E.g. (This is applied to every column of RDM)
Input vector: [3,5,4,2] 
Output 1st argsort: [3,0,2,1]
Output 2nd argsort: [1,3,2,0]
2 is the 1st NN, 3 is the 2nd, etc...

OUTPUT:
- ranks: np.ndarray (N,N) -> Interpretable columns, the ith element in column j 
is the distance rank of point i in the neighborhood of j.
- kmins: nd.ndarray (k, N) -> The indices of the k nearest neighbors (rows) for each points (cols)
"""
class InformationImbalance(RSA):
    def __init__(self, 
            signal_RDM_metric: str, 
            model_RDM_metric: str, 
            k: int = 1,
            RSA_metric: str='correlation', 
            signal_RDM: np.ndarray = None, 
            model_RDM: np.ndarray = None,
            ):
        self.k = k
        super().__init__(signal_RDM_metric, model_RDM_metric, RSA_metric, signal_RDM, model_RDM)
        
    # --- GETTERS ---
    def get_distance_ranks(self, RDM_type: str):
        return getattr(self, f"{RDM_type}_distance_ranks")
    # EOF    
    def get_kmins_idx(self, RDM_type: str):
        return getattr(self, f"{RDM_type}_kmins_idx")
    # EOF    
    def get_II(self, II_type: str):
        return getattr(self, f"II_{II_type}")
    # EOF    

    # --- SETTERS ---
    def set_distance_ranks_and_kmins(self, distance_ranks: np.ndarray[int], kmins_idx: np.ndarray[int], RDM_type: str):
        self.N = distance_ranks.shape[0]
        setattr(self, "{RDM_type}_distance_ranks", distance_ranks)
        setattr(self, "{RDM_type}_kmins_idx", kmins_idx)
    # EOF

    # --- OTHER FUNCTIONS ---
    def compute_distance_ranks(self, RDM_type: str):
        RDM = squareform(self.get_RDM(RDM_type))
        self.N = RDM.shape[0] # include it also in the setter
        np.fill_diagonal(RDM, np.inf)
        order = np.argsort(RDM, axis=0)
        # stores the indices of the k mins so that we don't have to compute the argmin later
        kmins = order[:self.k, :]
        setattr(self, f"{RDM_type}_kmins_idx", kmins) 
        ranks = np.argsort(order, axis=0)
        ranks = ranks + 2*np.eye(ranks.shape[0])
        setattr(self, f"{RDM_type}_distance_ranks", ranks)
        return ranks, kmins
    # EOF
    def compute_both_distance_ranks(self):
        ranks_signal, mins_signal = self.compute_distance_ranks("signal")
        ranks_model, mins_model = self.compute_distance_ranks("model")
        return ranks_signal, mins_signal, ranks_model, mins_model
    # EOF
    def compute_II(self, II_type: str):
        if II_type == 'A2B':
            conditioning_var = 'signal'
            conditioned_var = 'model'
        elif II_type == 'B2A':
            conditioning_var = 'model'
            conditioned_var = 'signal'
        # end if II_type == 'A2B':
        conditioning_mins = getattr(self, f"{conditioning_var}_kmins_idx")
        to_be_conditioned_ranks = getattr(self, f"{conditioned_var}_distance_ranks")
        conditioned_ranks = np.take_along_axis(to_be_conditioned_ranks, conditioning_mins, axis=0)
        II = (2/(self.N**2 * self.k))*np.sum(conditioned_ranks)
        setattr(self, f"II_{II_type}", II)
        return II
    # EOF
    def compute_both_II(self):
        II_A2B = self.compute_II('A2B')
        II_B2A = self.compute_II('B2A')
        return II_A2B, II_B2A
    # EOF
# EOC



"""
compare_similarity_metrics
Compute Information Imbalance (II) between two similarity metrics to compute RDMs
INPUT
- data : np.ndarray (D, N) -> Input data used to compute both RDMs.
- metric1, metric2 : str -> Similarity / distance metrics for signal and model RDMs.
- k : int -> Number of nearest neighbors used for conditioning.

OUTPUT:
- ii_obj : InformationImbalance -> Initialized and fully computed InformationImbalance object.
- ii_A2B : float -> Information Imbalance from metric1 to metric2.
- ii_B2A : float -> Information Imbalance from metric2 to metric1.
"""
def compare_similarity_metrics(data: np.ndarray, metric1: str, metric2: str, k: int): 
    ii_obj = InformationImbalance(metric1, metric2, k)
    ii_obj.compute_RDM(data, "signal")
    ii_obj.compute_RDM(data, "model")
    ii_obj.compute_both_distance_ranks()
    ii_A2B, ii_B2A = ii_obj.compute_both_II()
    return ii_obj, ii_A2B, ii_B2A
# EOF

class dynInformationImbalance(InformationImbalance, dRSA):
    def __init__(self, 
        signal_RDM_metric: str, 
        model_RDM_metric: str, 
        k: int = 1,
        RSA_metric: str='correlation', 
        signal_RDM: np.ndarray = None, 
        model_RDM: np.ndarray = None,
        signal_RDM_timeseries: TimeSeries = None, 
        model_RDM_timeseries: TimeSeries = None,
        model_RDM_static: np.ndarray = None,
        ):
            
        super().__init__(signal_RDM_metric, model_RDM_metric, k=k, RSA_metric=RSA_metric)
        self.signal_RDM_timeseries = signal_RDM_timeseries
        self.model_RDM_timeseries = model_RDM_timeseries
        self.model_RDM_static = model_RDM_static

    def compute_distance_ranks_dyn(self, RDM_t): # doesn't overrides the previous method because I added _dyn
        """ 
        auxiliary function for compute_distance_ranks_timeseries to compute the distance ranks without assigning
        any new attribute (useful for the loop)
        """
        RDM = squareform(RDM_t)
        # end if RDM_type is not None:
        self.N = RDM.shape[0] # include it also in the setter
        np.fill_diagonal(RDM, np.inf)
        order = np.argsort(RDM, axis=0)
        # stores the indices of the k mins so that we don't have to compute the argmin later
        kmins = order[:self.k, :]
        ranks = np.argsort(order, axis=0)
        ranks = ranks + 2*np.eye(ranks.shape[0])
        return ranks, kmins
    # EOF

    def compute_distance_ranks_timeseries(self, RDM_type): 
        """
        preceded by compute_RDM_timeseries from dRSA, computes the ranks (ranks_ts)
        and NNs for the target RDM timeseries
        """
        ranks_ts = []
        kmins_ts = []
        for RDM_t in self.get_RDM_timeseries(RDM_type):
                ranks, kmins = self.compute_distance_ranks_dyn(RDM_t)
                ranks_ts.append(ranks)
                kmins_ts.append(kmins)
        fs = self.get_RDM_timeseries(RDM_type).get_fs()
        ranks_ts = TimeSeries(ranks_ts, fs)
        kmins_ts = TimeSeries(kmins_ts, fs)
        setattr(self, f"{RDM_type}_distance_ranks_timeseries", ranks_ts)
        setattr(self, f"{RDM_type}_kmins_idx_timeseries", kmins_ts) 
    # EOF

    def compute_both_distance_ranks_timeseries(self): 
        """
        repeats compute_distane_ranks_timeseries twice
        """
        self.compute_distance_ranks_timeseries("signal")
        self.compute_distance_ranks_timeseries("model")

    def compute_dynII(self, II_type: str): # ADD the possibility to slice, accelerate the nested for loops with numba
        """
        computes the full lagged matrix of II. It's always (conditioning_var x conditioned_var), so pay attention 
        when comparing them
        """
        if II_type == 'A2B':
            conditioning_var = 'signal'
            conditioned_var = 'model'
        elif II_type == 'B2A':
            conditioning_var = 'model'
            conditioned_var = 'signal'
        # end if II_type == 'A2B':
        conditioning_mins_t = getattr(self, f"{conditioning_var}_kmins_idx_timeseries")
        to_be_conditioned_ranks_t = getattr(self, f"{conditioned_var}_distance_ranks_timeseries")
        dynII_mat = np.zeros((len(conditioning_mins_t), len(to_be_conditioned_ranks_t)))
        for i, mins in enumerate(conditioning_mins_t):
            for j, ranks in enumerate(to_be_conditioned_ranks_t):
                conditioned_ranks = np.take_along_axis(ranks, mins, axis=0)
                II = (2/(self.N**2 * self.k))*np.sum(conditioned_ranks)
                dynII_mat[i, j] = II
        setattr(self, f"dynII_{II_type}", dynII_mat)
        return dynII_mat
    # EOF

    def compute_both_dynII(self):
        """
        repeats compute_dynII in both directions
        """
        dynII_mat_A2B = self.compute_dynII('A2B')
        dynII_mat_B2A = self.compute_dynII('B2A')
        return dynII_mat_A2B, dynII_mat_B2A
    # EOF
    
    def compute_static_dynII(self, II_type):
        """
        computes the static_dynII (static model, dynamic signal)
        A is always the signal, but A_t in A2B is the signal's NNs, in
        B2A is the signal's ranks and viceversa for B. 
        """
        if II_type == 'A2B':
            A_t = getattr(self, f"signal_kmins_idx_timeseries")
            B = getattr(self, f"model_distance_ranks")
        elif II_type == 'B2A':
            A_t = getattr(self, f"signal_distance_ranks_timeseries")
            B = getattr(self, f"model_kmins_idx")
        # end if II_type == 'A2B':
        static_dynII = []
        for A in A_t:
            if II_type == 'A2B':
                conditioned_ranks = np.take_along_axis(B, A, axis=0) # ranks_B | mins_A = signal conditions model
            elif II_type == 'B2A':
                conditioned_ranks = np.take_along_axis(A, B, axis=0) # ranks_A | mins_B = model conditions signal
            # end if 'A2B':
            II = (2/(self.N**2 * self.k))*np.sum(conditioned_ranks)
            static_dynII.append(II)
        # end for ranks in to_be_conditioned_ranks_t:
        fs = A_t.get_fs()
        static_dynII = TimeSeries(static_dynII, fs)
        setattr(self, f"static_dynII_{II_type}", static_dynII)
        return static_dynII
    # EOF

    def compute_both_static_dynII(self):
        """
        computes the static_dynII in both senses
        """
        static_dynII_A2B = self.compute_static_dynII('A2B')
        static_dynII_B2A = self.compute_static_dynII('B2A')
        return static_dynII_A2B, static_dynII_B2A
    # EOF
# EOC


