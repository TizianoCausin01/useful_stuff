import sys, os
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLassoCV, MultiTaskElasticNetCV
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

sys.path.append("../..")
from useful_stuff.general_utils import print_wise, shift_concatenate_xy, get_lags, TimeSeries

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


class linear_encoding:
    """
    __init__
    Initializes the linear encoding model specifying regression type, cross-validation scheme and scoring method.

    INPUT:
        - regression_type: str -> type of regression model to use (e.g. ridge, lasso)
        - cv_type: str -> type of cross-validation strategy
        - alphas: np.ndarray(float) -> regularization parameters for regression
        - score_type: str -> metric used to evaluate predictions ('r2' or 'corr')
        - n_splits: int -> number of cross-validation folds
        - shuffle: bool -> whether to shuffle samples before splitting
        - **kwargs: dict -> additional parameters passed to the regression model

    OUTPUT:
        - None
    """
    def __init__(self, regression_type: str, cv_type: str, alphas: np.ndarray=np.logspace(-6, 3, 10), score_type: str='r2', n_splits: int=5, shuffle: bool=True, **kwargs):
        self.regression_type = regression_type 
        self.alphas = alphas
        self.regression_obj = choose_regression_type(self.regression_type, alphas=self.alphas, **kwargs)
        self.cv_type = cv_type
        self.cv_obj = choose_CV_type(cv_type, n_splits=n_splits, shuffle=shuffle)
        self.additional_args = kwargs
        self.score_type = score_type
    # EOF
    # --- GETTERS ---
    def get_regression_type(self):
        return self.regression_type
    # EOF
    def get_regression_obj(self):
        return self.regression_obj
    # EOF
    def get_alphas(self):
        return self.alphas
    # EOF
    def get_cv_type(self):
        return self.cv_type
    # EOF
    def get_cv_obj(self):
        return self.cv_obj
    # EOF
    def get_weights(self):
        return self.regression_obj.coef_
    # EOF
    def get_intercept(self):
        return self.regression_obj.intercept_
    # EOF
    def get_score_vals(self):
        return self.score_vals
    # EOF
    def get_score_type(self):
        return self.score_type
    # EOF
    # --- SETTERS ---
    def set_regression_type(self, regression_type: str, alphas=None, **kwargs):
        self.regression_type = regression_type
        if alphas is not None:
            self.alphas = alphas
        # end if alphas is not None:
        if kwargs is not None:
            self.additional_args = kwargs
        # end if kwargs is not None:
        self.regression_obj = choose_regression_type(self.regression_type, alphas=self.alphas, **self.additional_args)
    # EOF
    def set_regression_obj(self):
        raise AttributeError("Pass through set_regression_type to set a new regression object")
    # EOF
    def set_alphas(self):
        raise AttributeError("Pass through set_regression_type to set a new set of alphas")
    # EOF
    def set_cv_type(self, cv_type, n_splits=5, shuffle=True):
        self.cv_type = cv_type
        self.cv_obj = choose_CV_type(cv_type, n_splits=n_splits, shuffle=shuffle)
    # EOF
    def set_cv_obj(self):
        raise AttributeError("Pass through set_cv_type to set a new cv_obj")
    # EOF

    # --- OTHER METHODS ---
    """
    fit
    Fits the regression model to the training data.

    INPUT:
        - X_train: np.ndarray(float) -> (features x samples) or (samples x features)
        - Y_train: np.ndarray(float) -> (outputs x samples) or (samples x outputs)
        - transpose: bool -> if True transposes inputs to (samples x features)

    OUTPUT:
        - weights: np.ndarray(float) -> regression coefficients
        - intercept: np.ndarray(float) or float -> intercept parameter
    """
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, transpose=True):
        if transpose:
            X_train = np.ascontiguousarray(X_train.T)
            Y_train = np.ascontiguousarray(Y_train.T)
        # end if transpose:
        self.regression_obj.fit(X_train, Y_train)
        return self.get_weights(), self.get_intercept()
    # EOF
    """
    predict
    Generates predictions from the fitted regression model.

    INPUT:
        - X: np.ndarray(float) -> input data
        - transpose: bool -> whether to transpose input to (samples x features)
        - transpose_output: bool -> whether to return predictions as (outputs x samples)

    OUTPUT:
        - y_hat: np.ndarray(float) -> predicted outputs
    """
    def predict(self, X: np.ndarray, transpose=True, transpose_output=True):
        if transpose:
            X = np.ascontiguousarray(X.T)
        # end if transpose:
        y_hat = self.regression_obj.predict(X)
        if transpose_output:
            y_hat = np.ascontiguousarray(y_hat.T)
        # if transpose_output:
        return y_hat
    # EOF
    """
    score
    Evaluates model predictions on a test dataset.

    INPUT:
        - X_test: np.ndarray(float) -> test input data
        - Y_test: np.ndarray(float) -> ground truth outputs
        - y_hat: np.ndarray(float) -> optional precomputed predictions
        - transpose: bool -> whether to transpose inputs to (samples x features)
        - transpose_prediction: bool -> whether to transpose predictions

    OUTPUT:
        - score: np.ndarray(float) -> prediction score for each output
    """
    def score(self, X_test: np.ndarray, Y_test: np.ndarray, y_hat=None, transpose=True, transpose_prediction=True):
        if transpose:
            X_test = np.ascontiguousarray(X_test.T)
            Y_test = np.ascontiguousarray(Y_test.T)
        # end if transpose:
        if y_hat is None:
            y_hat = self.predict(X_test, transpose=False, transpose_output=False)
        else:
            if transpose_prediction:
                y_hat = np.ascontiguousarray(y_hat.T)
            # end if transpose_prediction:
        # end if y_hat is None:
        if self.score_type=='r2':
            score = r2_score(Y_test, y_hat, multioutput="raw_values")
        elif self.score_type=="corr": 
            score = evaluate_prediction_corr(Y_test, y_hat)
        # end if score_type=='r2':
        self.score_vals = score
        return score
    # EOF
    """
    crossvalidate
    Performs cross-validation to evaluate model performance.

    INPUT:
        - X: np.ndarray(float) -> input data
        - Y: np.ndarray(float) -> target outputs
        - transpose: bool -> whether to transpose inputs to (samples x features)

    OUTPUT:
        - score: np.ndarray(float) -> average cross-validation score across folds
    """
    def crossvalidate(self, X, Y, transpose=True):
        if transpose:
            X = np.ascontiguousarray(X.T)
            Y = np.ascontiguousarray(Y.T)
        # end if transpose:
        counter = 0
        score = None
    
        for train_idx, test_idx in self.get_cv_obj().split(X):
            X_train, Y_train = X[train_idx, :], Y[train_idx, :]
            X_test, Y_test = X[test_idx, :], Y[test_idx, :]
            self.fit(X_train, Y_train, transpose=False)
            counter+=1
            if score is None:
                score = self.score(X_test, Y_test, transpose=False)
            else:
                score += self.score(X_test, Y_test, transpose=False)
            # end if score is None:
        # end for train_idx, test_idx in self.get_cv_obj().split(X):
        score = score/counter
        self.score_vals = score
        return score
    # EOC


class dyn_linear_encoding(linear_encoding):
    """
    __init__
    Initializes the dynamic linear encoding model with lag parameters and inherits regression settings.

    INPUT:
        - regression_type: str -> type of regression (e.g. ridge, lasso)
        - cv_type: str -> cross-validation type
        - max_lag: int -> maximum lag for dynamic modeling
        - symmetric: bool -> whether to use symmetric lag range
        - alphas: np.ndarray(float) -> regularization parameters
        - score_type: str -> scoring metric ('r2' or 'corr')
        - n_splits: int -> cross-validation folds
        - shuffle: bool -> whether to shuffle samples
        - **kwargs: dict -> additional regression arguments

    OUTPUT:
        - None
    """
    def __init__(self, regression_type, cv_type, max_lag: int, symmetric: bool=False, alphas = np.logspace(-6, 3, 10), score_type = 'r2', n_splits = 5, shuffle = True, **kwargs):
        super().__init__(regression_type, cv_type, alphas, score_type, n_splits, shuffle, **kwargs)
        self.max_lag = max_lag
        self.symmetric = symmetric
    # EOF
    # --- GETTERS ---
    def get_max_lag(self):
        return self.max_lag
    # EOF
    def get_symmetric(self):
        return self.symmetric
    # EOF
    def get_weights_dyn(self):
        return self.weights_dyn
    # EOF
    def get_intercepts_dyn(self):
        return self.intercepts_dyn
    # EOF
    # --- SETTERS ---
    def set_max_lag(self, max_lag: int):
        self.max_lag = max_lag
    # EOF
    def set_symmetric(self, symmetric: bool):
        self.symmetric = symmetric
    # EOF
    def set_weights_dyn(self, weights: TimeSeries):
        self.weights_dyn = weights
    # EOF
    def set_intercepts_dyn(self, intercepts: TimeSeries):
        self.intercepts_dyn = intercepts
    # EOF

    # --- OTHER METHODS ---
    # --- STATIC-DYNAMIC (= model is static, signal is dynamic)
    """
    fit_static_dyn
    Fits regression model on static features and dynamic outputs.

    INPUT:
        - X_train: np.ndarray(float) -> static features
        - Y_train: TimeSeries -> dynamic outputs (features x time)
        - transpose: bool -> whether to transpose input arrays

    OUTPUT:
        - weights_dyn: TimeSeries -> fitted regression weights per timepoint
        - intercepts_dyn: TimeSeries -> fitted intercepts per timepoint
    """
    def fit_static_dyn(self, X_train: np.ndarray, Y_train: TimeSeries, transpose=True):
        if transpose:
            X_train = np.ascontiguousarray(X_train.T)
        # end if transpose:
        weights_dyn = []
        intercepts_dyn = []
        for y_t in Y_train:
            if transpose:
                y_t = np.ascontiguousarray(y_t.T)
            # end if transpose:
            w_t, w0_t = self.fit(X_train, y_t, transpose=False)
            weights_dyn.append(w_t)
            intercepts_dyn.append(w0_t)
        weights_dyn = TimeSeries(weights_dyn, Y_train.get_fs())
        intercepts_dyn = TimeSeries(intercepts_dyn, Y_train.get_fs())
        self.weights_dyn = weights_dyn
        self.intercepts_dyn = intercepts_dyn
        return weights_dyn, intercepts_dyn
    # EOF
    """
    predict_static_dyn
    Generates predictions using static-dynamic fitted weights.

    INPUT:
        - X: np.ndarray(float) -> static features
        - transpose: bool -> whether to transpose input
        - transpose_output: bool -> whether to transpose output to (outputs x time)

    OUTPUT:
        - y_hat_dyn: TimeSeries -> predicted dynamic outputs
    """
    def predict_static_dyn(self, X: np.ndarray, transpose=True, transpose_output=True):
        y_hat_dyn = []
        if transpose:
            X = X.T
        # end if transpose
        for w, i in zip(self.get_weights_dyn(), self.get_intercepts_dyn()):
            # if len(w.shape) == 2: # transpose in case you had to predict multiple outputs
            w = w.T
            # end if len(w.shape) == 2:
            y_hat =  X @ w + i
            if transpose_output:
                y_hat = y_hat.T
            # end if transpose_output:
            y_hat_dyn.append(y_hat)
        # end for w, i in zip(self.get_weights_dyn(), self.get_weights_dyn()):
        y_hat_dyn = TimeSeries(y_hat_dyn, self.get_weights_dyn().get_fs())
        return y_hat_dyn
    # EOF
    """
    score_static_dyn
    Evaluates static-dynamic predictions using R^2 or correlation.

    INPUT:
        - X_test: np.ndarray(float) -> test features
        - Y_test: TimeSeries -> ground truth outputs
        - y_hat: TimeSeries -> optional precomputed predictions
        - transpose: bool -> whether to transpose inputs
        - transpose_prediction: bool -> whether to transpose predictions

    OUTPUT:
        - score_dyn: np.ndarray(float) -> score per timepoint
    """
    def score_static_dyn(self, X_test: np.ndarray, Y_test: TimeSeries, y_hat=None, transpose=True, transpose_prediction=True):
        if transpose:
            X_test = np.ascontiguousarray(X_test.T)
            # Y_test = np.ascontiguousarray(Y_test.T)
        # end if transpose:
        if y_hat is None:
            y_hat = self.predict_static_dyn(X_test, transpose=False, transpose_output=False)
        else:
            if transpose_prediction:
                y_hat = TimeSeries([np.ascontiguousarray(y_t.T) for y_t in y_hat], y_hat.get_fs())
            # end if transpose_prediction:
        # end if y_hat is None:
        score_dyn = []
        for y_t, y_hat_t in zip(Y_test, y_hat):
            if self.score_type=='r2':
                score = r2_score(y_t, y_hat_t, multioutput="raw_values")
            elif self.score_type=="corr": 
                score = evaluate_prediction_corr(y_t, y_hat_t)
            # end if score_type=='r2':
            score_dyn.append(score)
        # end for y_t, y_hat_t in zip(Y_test, y_hat):
        time_score = np.stack(time_score, axis=-1) # time in the 0th or 1st axis depending on the evaluation used (corr has 1 feat, MSE has D)
        if len(time_score.shape)==1: # appends a dimension if the score is correlation
            time_score = time_score[np.newaxis, :]
        # end if len(time_score.shape)==1:
        self.score_vals_dyn = TimeSeries(score_dyn, Y_test.get_fs())
        return score_dyn
    # EOF
    """
    crossvalidate_static_dyn
    Performs cross-validation on static-dynamic model.

    INPUT:
        - X: np.ndarray(float) -> static features
        - Y: TimeSeries -> dynamic outputs
        - transpose: bool -> whether to transpose inputs

    OUTPUT:
        - TimeSeries -> cross-validated score over time
    """
    def crossvalidate_static_dyn(self, X: np.ndarray, Y: TimeSeries, transpose=True):
        time_score = []
        for y_t in Y:
            score = self.crossvalidate(X, y_t, transpose=transpose)
            time_score.append(score)
        # end for y_t in Y:
        time_score = np.stack(time_score, axis=-1) # time in the 0th or 1st axis depending on the evaluation used (corr has 1 feat, MSE has D)
        if len(time_score.shape)==1: # appends a dimension if the score is correlation
            time_score = time_score[np.newaxis, :]
        # end if len(time_score.shape)==1:
        return TimeSeries(time_score, Y.get_fs())
    # EOF

    # --- TIME GENERAL (= shared weights for the same lag across datapoints)
    """
    fit_general_dyn
    Fits regression model using shared weights across datapoints for each lag.

    INPUT:
        - X_train: TimeSeries -> input features over time
        - Y_train: TimeSeries -> output features over time
        - transpose: bool -> whether to transpose inputs

    OUTPUT:
        - weights_dyn: TimeSeries -> regression weights per lag
        - intercepts_dyn: TimeSeries -> intercepts per lag
    """
    def fit_general_dyn(self, X_train: TimeSeries, Y_train: TimeSeries, transpose=True):
        weights_dyn = []
        intercepts_dyn = []
        lags_range = get_lags(self.max_lag, self.symmetric)
        for tau in lags_range:
            x_shifted, y_shifted = shift_concatenate_xy(X_train, Y_train, tau, transpose=transpose) 
            w_t, w0_t = self.fit(x_shifted, y_shifted, transpose=False)
            weights_dyn.append(w_t)
            intercepts_dyn.append(w0_t)
        # end for tau in lags_range:
        weights_dyn = TimeSeries(weights_dyn, Y_train.get_fs())
        intercepts_dyn = TimeSeries(intercepts_dyn, Y_train.get_fs())
        self.weights_dyn = weights_dyn
        self.intercepts_dyn = intercepts_dyn
        return weights_dyn, intercepts_dyn
    # EOF
    """
    predict_general_dyn
    Predicts outputs for general dynamic model using shared weights.

    INPUT:
        - X_t: np.ndarray(float) -> features at a given timepoint
        - transpose: bool -> whether to transpose inputs
        - transpose_output: bool -> whether to transpose outputs

    OUTPUT:
        - TimeSeries -> predicted outputs over time
    """
    def predict_general_dyn(self, X_t: np.ndarray, transpose=True, transpose_output=True): # given a timepoint in the model, I want to predict the neural activity before and after it
        # don't accept X as a TimeSeries because the shift concatenate would yield a not so interpretable output otherwise
        return self.predict_static_dyn(X_t, transpose=transpose, transpose_output=transpose_output) # it is the same as the static version, only in this case the weights start from before the model at t
    # EOF
    """
    score_general_dyn
    Evaluates general dynamic model predictions across lags.

    INPUT:
        - X_test: TimeSeries -> input features
        - Y_test: TimeSeries -> ground truth outputs
        - y_hat: TimeSeries -> optional precomputed predictions
        - transpose_prediction: bool -> whether to transpose predictions

    OUTPUT:
        - TimeSeries -> score per lag
    """
    def score_general_dyn(self, X_test: TimeSeries, Y_test: TimeSeries, y_hat=None, transpose_prediction=True):
        lags_range = get_lags(self.max_lag, self.symmetric)
        w = self.get_weights_dyn()       
        i = self.get_intercepts_dyn()       
        time_score = []
        for idx, tau in enumerate(lags_range):
            x_shifted, y_shifted = shift_concatenate_xy(X_test, Y_test, tau, transpose=True) # transpose=True because the TimeSeries will be DxN for sure
            w_t = w[idx]
            i_t = i[idx]
            if y_hat is None:
                w_t = w_t.T
                y_hat_t =  x_shifted @ w_t + i_t
            else:
                y_hat_t = y_hat[idx]
            # end if y_hat is None:
            if self.score_type=='r2':
                score = r2_score(y_shifted, y_hat_t, multioutput="raw_values")
            elif self.score_type=="corr": 
                score = evaluate_prediction_corr(y_shifted, y_hat_t)
            # end if score_type=='r2':
            time_score.append(score)
            # end for y_t, y_hat_t in zip(Y_test, y_hat):
        time_score = np.stack(time_score, axis=-1) # time in the 0th or 1st axis depending on the evaluation used (corr has 1 feat, MSE has D)
        if len(time_score.shape)==1: # appends a dimension if the score is correlation
            time_score = time_score[np.newaxis, :]
        # end if len(time_score.shape)==1:
        self.score_vals_dyn = TimeSeries(time_score, Y_test.get_fs())
        return self.score_vals_dyn
    # EOF
    """
    crossvalidate_general_dyn
    Performs cross-validation on general dynamic model across lags.

    INPUT:
        - X: TimeSeries -> input features
        - Y: TimeSeries -> output features

    OUTPUT:
        - TimeSeries -> cross-validated score per lag
    """
    def crossvalidate_general_dyn(self, X: TimeSeries, Y: TimeSeries):
        if len(X) != len(Y):
            raise IndexError("The duration of X doesn't match the duration of Y")
        
        if X.get_fs() != Y.get_fs():
            raise ValueError("The two TimeSeries have different sampling frequencies: {X.get_fs()=}, {Y.get_fs()=}")
        # end if x_fs != y_fs:
        lags_range = get_lags(self.max_lag, self.symmetric)
        time_score = []
        for tau in lags_range:
            x_shifted, y_shifted = shift_concatenate_xy(X, Y, tau, transpose=True) # transpose=True because the TimeSeries will be DxN for sure
            score = self.crossvalidate(x_shifted, y_shifted, transpose=False)
            time_score.append(score)
        # end for y_t in Y:
        time_score = np.stack(time_score, axis=-1) # time in the 0th or 1st axis depending on the evaluation used (corr has 1 feat, MSE has D)
        if len(time_score.shape)==1: # appends a dimension if the score is correlation
            time_score = time_score[np.newaxis, :]
        # end if len(time_score.shape)==1:
        self.score_vals_dyn = TimeSeries(time_score, Y.get_fs())
        return self.score_vals_dyn
    # EOF


    """
    pointwise_regress_out
    Regress out X from Y using a linear model and return residual time series.

    INPUT:
        - X: TimeSeries -> Regressor time series (e.g., eye movements).
        - Y: TimeSeries -> Target time series (e.g., MEG signals).
        - regression_type: str | None (default=None) -> Type of regression model to use.
            If provided, temporarily overrides the current regression setting.
        - switch_back: bool (default=True) -> Whether to restore the original regression
            type after computation.

    OUTPUT:
        - TimeSeries -> Residual signal after regressing out X from Y, with same
            sampling frequency as input.

    NOTES:
        - X and Y must have the same sampling frequency.
        - Signals are truncated to the minimum shared number of timepoints.
        - Regression is fit once over all timepoints (no cross-validation).
        - Output corresponds to: Y_clean = Y - Ŷ.
    """
    def pointwise_regress_out(self, X: TimeSeries, Y: TimeSeries, regression_type: str=None, switch_back: bool=True):
        if len(X) != len(Y):
            raise IndexError("The duration of X doesn't match the duration of Y")
        
        if X.get_fs() != Y.get_fs():
            raise ValueError("The two TimeSeries have different sampling frequencies: {X.get_fs()=}, {Y.get_fs()=}")
        # end if x_fs != y_fs:
        if regression_type is not None:
            old_regression_type = self.get_regression_type()
            self.set_regression_type(regression_type)
            print_wise(f"Switching to {self.get_regression_obj()}")
        # end if regression_type is not None:
        X = np.squeeze(X.get_array())
        y_fs = Y.get_fs()
        Y = np.squeeze(Y.get_array())
        self.fit(X, Y)
        y_hat = self.predict(X)
        y_regress_out = Y - y_hat
        y_regress_out = TimeSeries(y_regress_out, y_fs)
        if switch_back:
            if regression_type is not None:
                self.set_regression_type(old_regression_type)
                print_wise(f"Switching back to {self.get_regression_obj()}")
            # end if regression_type is not None:
        # end if switch_back:
        return y_regress_out
    # EOF

    def delay_embed_PCR_regress_out(self, X: TimeSeries, Y: TimeSeries, delays_to_embed: tuple[2], PCs_to_keep=None, pad_mode='edge', crop_end=True, regression_type: str=None, switch_back: bool=True):
       embedded_X = X.delay_embeddings(delays_to_embed, pad_mode=pad_mode)
       D, N = embedded_X.shape()
       if PCs_to_keep is None: # if not specified, we keep half of the embedding dimension D or half of the datapoints N depending on which is the dominant dimension of the matrix
           PCs_to_keep = round(min(D, N) / 2)
       elif PCs_to_keep > min(D, N):
           raise ValueError(f"You can't have more PCs ({PCs_to_keep}) that datapoints ({N}) or features ({D})")
       # end if PCs_to_keep is None: 
       pca_obj = PCA(n_components=PCs_to_keep)
       embedded_X = embedded_X.get_array().T
       reduced_emb_X = pca_obj.fit_transform(embedded_X)
       reduced_emb_X = TimeSeries(reduced_emb_X.T, X.get_fs())
       reduced_emb_X.resample(Y.get_fs())
       if crop_end:
           min_timepts = min(len(reduced_emb_X),len(Y)) 
           reduced_emb_X = TimeSeries(reduced_emb_X.get_array()[:, :min_timepts, ...], reduced_emb_X.get_fs())
           Y = TimeSeries(Y.get_array()[:, :min_timepts, ...], Y.get_fs()) #TODO check if this is modifying Y in place
       # end if crop_end:
       y_regress_out = self.pointwise_regress_out(reduced_emb_X, Y, regression_type=regression_type, switch_back=switch_back)
       return y_regress_out, pca_obj
    # EOF

    # TODO --- TIME SPECIFIC (= each datapoint has its own weight - typically used if we short stimuli with a specific timecourse)

    # time-dependent
    # fit

    # predict

    # score

    # cross-validate
