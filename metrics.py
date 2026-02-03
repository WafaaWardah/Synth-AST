"""
metrics.py: Helper methods for evaluation metrics

Author: Wafaa Wardah
Date: December 12, 2024
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t
from scipy.optimize import Bounds, minimize
from scipy.optimize import curve_fit, OptimizeWarning


def calculate_rmse_star_old(y, y_hat, y_std, num_votes, d=1, debug=True):
    # Total number of samples
    N = y.shape[0]

    # Input validation
    if len(y) <= d or y.isna().any() or y_hat.isna().any() or y_std.isna().any() or num_votes.isna().any():
        if debug:
            print("Invalid inputs detected. Returning NaN.")
        return np.nan

    # Calculate confidence intervals (ci)
    ci = np.where(
        num_votes < 30,
        t.ppf(0.975, np.maximum(num_votes - 1, 1)) * y_std,
        1.96 * y_std
    )

    # Calculate errors
    error = y - y_hat

    # Handle missing CI values
    if np.isnan(ci).any():
        if debug:
            print("Missing CI values detected. Returning NaN.")
        return np.nan

    # Epsilon-insensitive error
    p_error = (np.abs(error) - ci).clip(lower=0)

    # Check degrees of freedom
    if (N - d) < 1:
        if debug:
            print("Not enough degrees of freedom. Returning NaN.")
        return np.nan

    # Calculate RMSE*
    rmse_star = np.sqrt(np.sum(p_error ** 2) / (N - d))

    # Debugging Outputs
    if debug:
        print(f"Error: {error}")
        print(f"Confidence Intervals (ci): {ci}")
        print(f"Epsilon-Insensitive Error (p_error): {p_error}")
        print(f"RMSE*: {rmse_star}")

    return rmse_star


def calculate_rmse_star(y, y_hat, ci95, d=1, debug=False):
    # Total number of samples
    N = y.shape[0]

    # Input validation
    if len(y) <= d or y.isna().any() or y_hat.isna().any() or ci95.isna().any():
        if debug:
            print("Invalid inputs detected. Returning NaN.")
        return np.nan

    # Calculate errors
    error = y - y_hat

    # Epsilon-insensitive error
    p_error = (np.abs(error) - ci95).clip(lower=0)

    # Check degrees of freedom
    if (N - d) < 1:
        if debug:
            print("Not enough degrees of freedom. Returning NaN.")
        return np.nan

    # Calculate RMSE*
    rmse_star = np.sqrt(np.sum(p_error ** 2) / (N - d))

    # Debugging Outputs
    if debug:
        print(f"Error: {error}")
        print(f"Confidence Intervals (ci95): {ci95}")
        print(f"Epsilon-Insensitive Error (p_error): {p_error}")
        print(f"RMSE*: {rmse_star}")

    return rmse_star


def calculate_condition_std(votes_per_file, num_votes):
    """
    Calculate pooled standard deviation for a condition based on ITU-T P.1401.
    """
    valid_entries = [(vote, n) for vote, n in zip(votes_per_file, num_votes) if n > 0]
    if not valid_entries:
        return np.nan
    votes_per_file, num_votes = zip(*valid_entries)

    weighted_variances = sum((n - 1) * vote ** 2 for vote, n in zip(votes_per_file, num_votes))
    total_degrees_of_freedom = sum(n - 1 for n in num_votes)
    return np.sqrt(weighted_variances / total_degrees_of_freedom) if total_degrees_of_freedom > 0 else np.nan


def calculate_ci(std, total_votes):
    """
    Calculate the confidence interval (ci) based on ITU-T P.1401.
    """
    if np.isnan(std) or total_votes <= 1:
        return np.nan  # Not enough votes for meaningful CI
    
    if total_votes < 30:
        t_value = t.ppf(0.975, df=total_votes - 1)
        ci = t_value * (std / np.sqrt(total_votes))
    else:
        ci = 1.96 * (std / np.sqrt(total_votes))
    
    return ci


def aggr_per_con_db(file_preds_df, dim):
    """
    Aggregation function for getting per condition predictions dynamically based on the specified `dim`.
    Also calculates standard deviation and 95% confidence intervals for the given dimension.
    """
    # Define dynamic column names
    dim_pred = f'{dim}_pred'
    dim_pred_map_1st = f'{dim}_pred_map_1st'
    dim_pred_map_3rd = f'{dim}_pred_map_3rd'

    # Build aggregation dynamically
    agg_dict = {
        'db': ('db', 'first'),
        'file_count': ('file_num', 'count'),
        'num_votes': ('num_votes', 'sum'),
        dim: (dim, 'mean'),  # Dynamically aggregate the target column (e.g., `mos`)
        dim_pred: (dim_pred, 'mean'),
        dim_pred_map_1st: (dim_pred_map_1st, 'mean'),
        dim_pred_map_3rd: (dim_pred_map_3rd, 'mean'),
    }

    # Perform aggregation
    grouped_df = file_preds_df.groupby('con_num').agg(**agg_dict).reset_index()

    # Calculate standard deviation dynamically for the given dimension
    grouped_df[f'{dim}_std'] = grouped_df['con_num'].map(
        lambda con_num: calculate_condition_std(
            file_preds_df[file_preds_df['con_num'] == con_num][dim].values,
            file_preds_df[file_preds_df['con_num'] == con_num]['num_votes'].values
        )
    )

    # Calculate 95% confidence interval dynamically for the given dimension
    grouped_df[f'{dim}_ci95'] = grouped_df.apply(
        lambda row: calculate_ci(row[f'{dim}_std'], row['num_votes']), axis=1
    )

    return grouped_df


def calculate_pcc(y_true, y_pred):
    # Check for valid entries (non-NaN values)
    valid_entries = (~y_true.isna()) & (~y_pred.isna())

    # Extract valid values
    y_true_valid = y_true[valid_entries]
    y_pred_valid = y_pred[valid_entries]

    # Check for constant input
    if len(y_true_valid) < 2 or len(y_pred_valid) < 2:
        return np.nan  # Not enough data for correlation
    if np.all(y_true_valid == y_true_valid.iloc[0]) or np.all(y_pred_valid == y_pred_valid.iloc[0]):
        return np.nan  # Constant input array; correlation undefined

    return pearsonr(y_true_valid, y_pred_valid)[0]


def calculate_rmse(actual, predicted, d=1): 
    '''Calculate RMSE with (N - d) denominator'''
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length.")

    N = len(actual)
    if N <= d:
        return np.nan  # Not enough data to calculate RMSE with given degrees of freedom

    Perror = actual - predicted
    rmse = np.sqrt(np.sum(Perror ** 2) / (N - d))
    return rmse


def apply_first_order_map(y, y_hat):
    '''Calculate the first-order (linear) mapping'''
    if y is None or y_hat is None or len(y) < 2:
        return np.nan
    if np.isnan(y).any() or np.isnan(y_hat).any():
        return np.nan
    if np.std(y) < 1e-5 or np.std(y_hat) < 1e-5:
        return np.nan
    
    def linear_map(y, a, b):
        return a + b * y

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(linear_map, y_hat, y)
    except RuntimeError:
        return np.nan 

    a, b = params
    y_mapped_first_order = a + b * y_hat
    return y_mapped_first_order


def calculate_third_order_map(y, y_hat):
    '''Calculate the third-order (polynomial) mapping - NOT MONOTONOUS!'''
    if np.isnan(y).any() or np.isnan(y_hat).any():
        return np.nan

    def poly_map(y, a, b, c, d):
        return a + b * y + c * y**2 + d * y**3
    
    # Fit a third-order polynomial model to find 'a', 'b', 'c', and 'd'
    params, _ = curve_fit(poly_map, y_hat, y)
    a, b, c, d = params
    
    y_mapped_third_order = a + b * y_hat + c * y_hat**2 + d * y_hat**3
    return y_mapped_third_order


def polynomial_map(y, a, b, c, d):
    '''Calculate the third-order monotonous (polynomial) mapping'''
    return a + b * y + c * y**2 + d * y**3


def objective(params, y, x):
    '''RMSE objective function to minimize'''
    a, b, c, d = params
    y_mapped = polynomial_map(y, a, b, c, d)
    rmse = np.sqrt(np.mean((x - y_mapped) ** 2))
    return rmse


def monotonicity_constraint(params, y_min, y_max):
    '''Derivative of the polynomial: b + 2*c*y + 3*d*y^2'''
    a, b, c, d = params
    # Ensure that this derivative is non-negative over the interval [y_min, y_max]
    y_values = np.linspace(y_min, y_max, 100)  # Sample values in the range
    derivative = b + 2 * c * y_values + 3 * d * y_values**2
    return np.min(derivative)  # Constraint that this must be >= 0


def apply_third_order_monotonic_map(x, y):
    '''Apply the third order monotonic map'''
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    
    # Check for low variance
    if np.std(x) < 1e-5 or np.std(y) < 1e-5:
        return np.nan

    # Initial guess for parameters
    initial_params = [0, 1, 0, 0]  # Start with an approximate linear map

    # Bounds for a, b, c, d (if needed)
    bounds = Bounds([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    # Define constraints for monotonicity over the range [y_min, y_max]
    y_min, y_max = np.min(y), np.max(y)
    constraints = {
        'type': 'ineq',
        'fun': lambda params: monotonicity_constraint(params, y_min, y_max)
    }

    # Perform the minimization with constraints
    result = minimize(objective, initial_params, args=(y, x), bounds=bounds, constraints=constraints)
    a, b, c, d = result.x
    y_mapped = polynomial_map(y, a, b, c, d) # Apply the mapping with the optimized parameters
    return y_mapped


def calc_metrics_db(df, dim, verbose=True):
    '''
    Calculates model performance metrics per database, per file and per condition.
    ''' 
    db_list = df['db'].unique().tolist()

    metrics = ['pcc', 'rmse', 'rmse_star', 
            'rmse_mapped_first_order', 'rmse_mapped_third_order', 
            'rmse_mapped_first_order_star', 'rmse_mapped_third_order_star']
    dimensions = [dim]
    columns = [f"{dim}_{metric}" for dim in dimensions for metric in metrics]
    columns = ['db'] + columns

    rows_file = []

    for db in db_list:

        db_df = df[df['db'] == db]

        if db_df.empty:
            print(f"Warning: No data found for DB: {db}")
            continue

        db_metrics = {col: np.nan for col in columns}
        db_metrics['db'] = db
      
        # Calculate PCC for each dimension
        for dim in dimensions:
            db_metrics[f"{dim}_pcc"] = calculate_pcc(db_df[f"{dim}"], db_df[f"{dim}_pred"])
    
        if verbose:

            for dim in dimensions:
                pcc_f = db_metrics.get(f"{dim}_pcc", "N/A")
                print(f"File {db} | {(dim).upper():<4} | PCC: {pcc_f:.4f}")
                 
        rows_file.append(pd.DataFrame([db_metrics]))

    metrics_df = pd.concat(rows_file, ignore_index=True).dropna(axis=1, how='all')

    return metrics_df



#-------------------------------------------------------------------------------------------
# Sample predictions df from NISQA prediction output
#df = pd.read_csv("/Users/wafaa/Code/psamd/ast_models/MODEL/misc/nisqa_preds_scores.csv")

# Calculate metrics
#metrics_p_file, metrics_p_con = calc_metrics_db(df, 'mos')
