# Process raw data for the ML models

import pandas as pd

# Target variable
target_col_idx = 1
y_raw = pd.read_csv(
    '../data/profile.txt', 
    sep='\t', 
    usecols=[target_col_idx],
    names=['valve condition'],  
    dtype={target_col_idx: int}) \
    .iloc[:,0]

y = (y_raw - 99).clip(lower=0)

# Explanatory variables
FS1 = pd.read_csv('../data/FS1.txt', sep='\t', names=[str(i) + '_FS1' for i in range(1, 601)], dtype=float)
PS2 = pd.read_csv('../data/PS2.txt', sep='\t',  names=[str(i) + '_PS2' for i in range(1, 6001)],dtype=float)

# Save preprocessed data
y.to_csv('../data/y.csv', index=False)
FS1.to_csv('../data/FS1.csv', index=False)
PS2.to_csv('../data/PS2.csv', index=False)
