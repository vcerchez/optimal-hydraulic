# Pipeline for the model training, evaluation and pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

# Read raw data
# Target variable
target_col_idx = 1
y_raw = pd.read_csv(
    'data/profile.txt', 
    sep='\t', 
    usecols=[target_col_idx],
    names=['valve condition'],  
    dtype={target_col_idx: int}) \
    .iloc[:,0]

y = (y_raw - 99).clip(lower=0)

# Explanatory variables
# We ignore FS1, perfect model can be trained with PS2 only.
X = pd.read_csv('data/PS2.txt', sep='\t', names=[str(i) + '_PS2' for i in range(1, 6001)], dtype=float)

# Data preprocessing
# Reduce time resolution of X(PS2)
def reduce_time_resolution(df, stride):
    if stride == 1:
        return df
    
    df_reduced = []
    for i in range(0, df.shape[1], stride):
        avg_cols = df.iloc[:, i:i+stride].mean(axis=1)
        avg_cols.name = df.iloc[:, i].name
        df_reduced.append(avg_cols)

    df_reduced = pd.concat(df_reduced, axis=1)

    return df_reduced

X = reduce_time_resolution(X, stride=500)

# Train and evaluate LR model
# TODO: CV hyperparameter tuning
scores = cross_validate(
    LogisticRegression(random_state=0, max_iter=300), 
    X, 
    y, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1), 
    scoring='accuracy', 
    return_train_score=True
    )

scores = pd.DataFrame(scores)[['train_score', 'test_score']]

# Train the model
clf = LogisticRegression(random_state=0, max_iter=300).fit(X, y)
