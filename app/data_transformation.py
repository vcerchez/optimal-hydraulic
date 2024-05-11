# Defines data transformation pipeline

import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def _extract_PS2(X: pd.DataFrame, ncols: int) -> pd.DataFrame:
    """Extract PS2 features.

    It is supposed that X is the result of concatenation (some tables, PS2) in this order.

    Args:
        X (pd.DataFrame): Original dataframe containing examples with all features.
        ncols (int): Number of PS2 columns.

    Returns:
        pd.DataFrame: Table with PS2 features only.
    """    
    return X.iloc[:, -ncols:]

def _reduce_time_resolution(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Reduce time resolution of data.

    Args:
        df (pd.DataFrame): Input data frame
        stride (int): Time resolution reduction coefficient.

    Returns:
        pd.DataFrame: Input data with reduced time resolution.
    """    
    if stride == 1:
        return df
    
    df_reduced = []
    for i in range(0, df.shape[1], stride):
        avg_cols = df.iloc[:, i:i+stride].mean(axis=1)
        # new column will have the name of the 1st column in the stride
        avg_cols.name = df.iloc[:, i].name
        df_reduced.append(avg_cols)

    df_reduced = pd.concat(df_reduced, axis=1)

    return df_reduced

def extract_PS2(ncols: int):
    return FunctionTransformer(_extract_PS2, check_inverse=False, kw_args=dict(ncols=ncols))

def reduce_time_resolution(stride: int):
    return FunctionTransformer(_reduce_time_resolution, check_inverse=False, kw_args=dict(stride=stride))
