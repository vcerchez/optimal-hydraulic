# Test app serving ML model requests

import numpy as np
import pandas as pd
import pytest
from data_transformation import *

# Actual raw data has dims: X (N, 6600), y (N,)

def test_PS2_extraction():
    """Test PS2 features extraction from complete feature vectors."""

    df1 = pd.DataFrame([[1, 2, 3], [8, 9, 10]], columns=list('ABC'))
    df2 = pd.DataFrame([[4, 5, 6, 7], [11, 12, 13, 14]], columns=list('ABCD'))
    
    X = pd.concat([df1, df2], axis=1)
    expected = df2
    
    actual = extract_PS2(ncols=df2.shape[1]).transform(X)
    
    pd.testing.assert_frame_equal(expected, actual)


def df():
    return pd.DataFrame(data = [[1, 2, 3, 4], [5, 6, 7, 8]], columns=list('ABCD'))

@pytest.mark.parametrize(
        "df, stride, expected", 
        [(df(), 1, df()), 
         (df(), 2, pd.DataFrame(data = [[(1+2)/2, (3+4)/2], [(5+6)/2, (7+8)/2]], columns=['A', 'C'])), 
         (df(), 3, pd.DataFrame(data = [[(1+2+3)/3, 4/1], [(5+6+7)/3, 8/1]], columns=['A', 'D'])), 
         (df(), 4, pd.DataFrame(data = [[(1+2+3+4)/4, ], [(5+6+7+8)/4, ]], columns=['A', ]))], 
         ids=[f'stride {i}' for i in [1, 2, 3, 4]])
def test_reduce_time_resolution(df, stride, expected):
    """Test time resolution reduction function."""
    actual = reduce_time_resolution(stride=stride).transform(df)
    
    pd.testing.assert_frame_equal(expected, actual)
