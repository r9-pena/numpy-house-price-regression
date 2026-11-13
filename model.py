"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    x = np.asarray(X.copy())
    means=np.nanmean(x,axis=0)
    means = np.nan_to_num(means,nan=0.0)
    idx = np.where(np.isnan(x))
    x[idx] = means[idx[1]]

    return x

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    x = np.asarray(X.copy())
    q1 = np.percentile(x,25,axis=0)
    q3 = np.percentile(x,75,axis=0)
    iqr = q3 - q1

    lower = q1 - k*iqr
    upper = q3 + k*iqr
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    x = np.asarray(X.copy())
    return np.clip(x,lower,upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    return (numerator/(denominator+eps))

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    x = np.asarray(X)
    c = np.asarray(col).T
    return np.column_stack([x,c])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    hot = np.asarray(labels.copy())
    n = len(hot)
    unique_val = np.unique(hot)
    c = len(unique_val)
    omatrix = np.zeros((n,c))
    omatrix[hot[:,None]==unique_val[None,:]] = 1.0
    return omatrix

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    x=np.asarray(X.copy())
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    std[std==0] = 1.0
    return mean,std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    x = np.asarray(X.copy())
    return (x-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    x = np.asarray(X.copy())
    row,col = np.shape(x)
    b = np.ones((row,1))
    return np.hstack((b,x))

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    idx = np.arange(n_samples)
    np.random.RandomState(seed).shuffle(idx)
    return idx

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    idx = np.asarray(indices.copy())
    p1 = int(n*train_ratio)
    p2 = int(p1 + n*val_ratio)
    # train = idx[:p1]
    return idx[0:p1],idx[p1:p2],idx[p2:]

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    x_sub = X[indices]
    y_sub = y[indices]
    return x_sub,y_sub

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    A = X.T@X
    b = X.T@y
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X@theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    mae = np.abs(y_true-y_pred)
    return np.mean(mae)

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    mse = (y_true-y_pred)**2
    mse = np.sqrt(np.mean(mse))
    return mse

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    res = np.sum((y_true-y_pred)**2)
    y_hat = np.mean(y_true)
    tot = np.sum((y_true-y_hat)**2)
    return (1.0-res/tot) if tot != 0 else 0

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r = y_true - y_pred
    r_mean = float(np.mean(r))
    std = float(np.std(r-r_mean))
    r_median = float(np.median(np.abs(r)))
    dic = {
        'mean':r_mean,
        'std':std,
        'median_abs':r_median
    }
    return dic

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    x = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(x,iqr_k)
    x = clip_columns(x, lower, upper)
    return x

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    rat = make_ratio_feature(X_num[:,ratio_num_idx],X_num[:,ratio_den_idx])
    x = append_column(X_num,rat)

    if cat_labels is not None:
        hot = one_hot_encode(cat_labels)
        x = append_column(x,hot)
    return x

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    dic = {}
    n_samples = X.shape[0]
    idx = make_shuffled_indices(n_samples,seed)
    train,val,test = partition_indices(idx, train_ratio,val_ratio) 
    dic['X_train'],dic['y_train'] = subset_xy(X,y,train)
    dic['X_val'],dic['y_val'] = subset_xy(X,y,val)
    dic['X_test'],dic['y_test'] = subset_xy(X,y,test)

    return dic

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    dic = splits.copy()
    mean, std = fit_standardizer(splits['X_train'])
    dic['X_train'] = add_bias_column(apply_standardizer(splits['X_train'],mean,std))
    dic['X_val'] = add_bias_column(apply_standardizer(splits['X_val'],mean,std))
    dic['X_test'] = add_bias_column(apply_standardizer(splits['X_test'],mean,std))

    return dic,mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    dic = {}
    dic['mae']=mean_absolute_error(y_true,y_pred)
    dic['rmse']=root_mean_squared_error(y_true,y_pred)
    dic['r2']=r_squared(y_true,y_pred)
    dic['residual_summary']=residual_summary(y_true,y_pred)
    return dic

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    dic={}
    x_clean = prepare_cleaned_features(X,iqr_k)
    x_ft = assemble_feature_matrix(x_clean,ratio_num_idx,ratio_den_idx,cat_labels)
    splits = make_train_val_test(x_ft,y,train_ratio,val_ratio,seed)
    std_split,_,_= standardize_and_add_bias(splits)
    theta = ols_fit(std_split['X_train'],std_split['y_train'])
    y_val = ols_predict(std_split['X_val'],theta)
    y_test = ols_predict(std_split['X_test'],theta)

    dic['val_metrics'] = evaluate_predictions(std_split['y_val'],y_val)
    dic['test_metrics'] = evaluate_predictions(std_split['y_test'],y_test)
    dic['theta']=theta
    dic['y_test']=std_split['y_test']
    dic['y_test_pred']=y_test

    return dic

