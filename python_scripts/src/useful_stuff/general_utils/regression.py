import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, LeaveOneOut

sys.path.append("../..")
from useful_stuff.general_utils import shift_concatenate_xy

"""
IdentitySplit
A simple cross-validation class that mimics the scikit-learn CV API but returns
identical train and test splits. Useful when training and testing on the full
dataset is desired, while retaining compatibility with CV-based pipelines.

INPUT (constructor):
    - shuffle: bool (default False) -> Whether to shuffle the indices before splitting.
    - random_state: int or None -> Random seed used only when shuffle=True.

METHODS:
    - split(X, y=None, groups=None) -> Yields (train_idx, test_idx) where both are identical arrays of indices.
    - get_n_splits(...) -> Returns 1, as only one split is produced.

OUTPUT:
    - A generator yielding one pair (idx, idx) of identical train/test indices.
"""
class IdentitySplit(BaseCrossValidator):
    
    def __init__(self, shuffle=False, random_state=None):
        self.shuffle = shuffle
        self.random_state = random_state
    # EOF

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        idx = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(idx)  # shuffle indices

        yield idx, idx
    # EOF
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1  # required by BaseCrossValidator
    # EOF

# EOC


"""
choose_CV_type
Defines the CV object.
INPUT:
    - cv_type: str -> the cross-validation type to use
    - n_splits: int -> how many splits in case of k-fold
    - shuffle: bool -> shuffle the data or not to improve generalization
OUTPUT:
    - CV: sklearn.model_selection._split.KFold (or the others) -> object to split dataset into train and test sets
"""
def choose_CV_type(cv_type, n_splits=5, shuffle=True):
    if cv_type == 'same':
        CV = IdentitySplit(shuffle=shuffle)
    elif cv_type == 'loo': # leave one-out
        CV = LeaveOneOut()
    elif cv_type == 'kf': # leave one-out
        CV = KFold(n_splits=n_splits, shuffle=shuffle)
    else:
        raise ValueError("cv_type must be 'same', 'loo', 'kf'")
    # end if cv_type == 'same':
    return CV
# EOF


"""
choose_regression_type
Defines the regression type.
INPUT:
    - regression_type: str -> the regression type
    - alphas: np.ndarray -> regularization parameter to search in
    - l1_ratio: list -> hyperparameter for elasticnet
    - **kwargs: dict -> in case you wanted to add more
OUTPUT:
    - regression_obj: sklearn.linear_model._base.LinearRegression (or other) -> the regression object to fit the data
"""
def choose_regression_type(regression_type, alphas=np.logspace(-6, 3, 10), l1_ratio=[0.1, 0.5, 0.9], **kwargs):
    if regression_type == 'lr': 
        regression_obj = LinearRegression(**kwargs)
    elif regression_type == 'ridge': 
        regression_obj = RidgeCV(alphas=alphas, **kwargs)
    elif regression_type == 'lasso': 
        regression_obj = MultiTaskLassoCV(alphas=alphas, max_iter=10000, **kwargs)
    elif regression_type == 'en': 
        regression_obj = MultiTaskElasticNetCV(alphas=alphas, max_iter=10000, l1_ratio=l1_ratio, **kwargs)
    else:
        raise ValueError("regression_type must be 'lr', 'ridge', 'lasso', 'en'")
    # end if regression_type == 'lr': 
    return regression_obj
# EOF


"""
evaluate_prediction_corr
Evaluates a trained regression model by computing the average correlation
between predicted and true signals across output dimensions.

INPUT:
    - test_x: np.ndarray (samples, features) -> Data used for prediction.
    - test_y: np.ndarray (samples, output_features) -> Ground-truth targets.
    - regression_obj: fitted sklearn estimator -> Model used to generate predictions.

OUTPUT:
    - avg_corr: float -> Mean Pearson correlation across output dimensions between y_hat and test_y.
"""
def evaluate_prediction_corr(Y_test, y_hat):
    if Y_test.shape[1]==1:
        raise ValueError("You can't run correlation with just 1 feature")
    # end if Y_test.shape[1]==1:
    corr_vals = np.array([np.corrcoef(y_hat[i, :], Y_test[i, :])[0, 1] for i in range(Y_test.shape[0])])
    if np.any(np.isnan(corr_vals)):
        Warning("We have NaNs in the correlation values")
    # end if np.any(np.isnan(corr_vals)):
    avg_corr = np.nanmean(corr_vals)
    return avg_corr
# EOF


"""
lagged_linear_regression
Performs lagged regression between two time-series arrays x (model) and y (neural signal)
across a range of temporal lags, using a chosen regression method and
cross-validation strategy.

For each lag tau:
    1. x and y are shifted and concatenated across trials.
    2. A regression model is trained using the chosen CV strategy.
    3. Predictive performance is evaluated using correlation.
    4. The mean correlation across CV folds is stored.

INPUT:
    - x: np.ndarray (features, timepoints, trials) -> Trials of the regressor time series.
    - y: np.ndarray (features, timepoints, trials) -> Trials of neural signal time series.
    - regression_type: str (default 'lr') -> type of regression: 'lr', 'ridge', 'lasso', 'en'.
    - alpha: float (default 0.0) -> regularization strength (used for ridge, lasso, elastic net).
    - cv_type: str (default 'same') -> Cross-validation type: 'same', 'loo', or 'kf'.
    - n_splits: int (default 5) -> Number of splits used for k-fold CV.
    - shuffle: bool (default True) -> Whether to shuffle data in CV.
    - max_lag: int (default 20) -> Maximum temporal lag.
    - symmetric: bool (default False)
          If True → compute only non-negative lags 0..max_lag.
          If False → compute symmetric lags from −max_lag..max_lag.

OUTPUT:
    - lr_list: list of floats -> Mean predictive correlation for each tested lag. Ordered according to the lag range used.
"""
def lagged_linear_regression(x, y, regression_type='lr', alpha=0.0, cv_type='same', n_splits=5, shuffle=True, max_lag=20, symmetric=False):
    lr_list = []
    if symmetric:
        lags_range = range(max_lag+1)
    else:
        lags_range = range(-max_lag, max_lag+1)
    # end if symmetric:

    for tau in lags_range:
        x_shifted_tot, y_shifted_tot = shift_concatenate_xy(x, y, tau)
        CV = choose_CV_type(cv_type, n_splits=n_splits, shuffle=shuffle)
        curr_lr = []
        for train_idx, test_idx in CV.split(x_shifted_tot):
            train_x, train_y = x_shifted_tot[train_idx, :], y_shifted_tot[train_idx, :]
            test_x, test_y = x_shifted_tot[test_idx, :], y_shifted_tot[test_idx, :]
            regression_obj = choose_regression_type(regression_type=regression_type, alpha=alpha)
            regression_obj = regression_obj.fit(train_x, train_y)
            #R2 = regression_obj.score(test_x, test_y) # later add correlation and see if add weights or something else
            avg_corr = evaluate_prediction(test_x, test_y, regression_obj)
            curr_lr.append(avg_corr)
        # end for train_idx, test_idx in CV.split(x_shifted_tot):
        lr_list.append(np.nanmean(curr_lr))  #x_t_shifted_tot, y_t_shifted_tot))
    # end for L in range(-max_lag, max_lag):    
    return lr_list
# EOF

def static_lagged_linear_regression(x, y, regression_type='lr', alpha=0.0, cv_type='same', n_splits=5, shuffle=True):
    lr_list = []
    x_shifted_tot = x.T
    for tau in range(y.shape[1]):
        y_shifted_tot = y[:,tau,:].T
        CV = choose_CV_type(cv_type, n_splits=n_splits, shuffle=shuffle)
        curr_lr = []
        for train_idx, test_idx in CV.split(x_shifted_tot):
            train_x, train_y = x_shifted_tot[train_idx, :], y_shifted_tot[train_idx, :]
            test_x, test_y = x_shifted_tot[test_idx, :], y_shifted_tot[test_idx, :]
            regression_obj = choose_regression_type(regression_type=regression_type, alpha=alpha)
            regression_obj = regression_obj.fit(train_x, train_y)
            avg_corr = evaluate_prediction(test_x, test_y, regression_obj)
            curr_lr.append(avg_corr)
        # end for train_idx, test_idx in CV.split(x_shifted_tot):
        lr_list.append(np.nanmean(curr_lr))  #x_t_shifted_tot, y_t_shifted_tot))
    # end for L in range(-max_lag, max_lag):    
    return lr_list
# EOF
