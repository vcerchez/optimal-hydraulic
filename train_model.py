# Pipeline for the model training, evaluation and pickle

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import joblib

from data_transformation import extract_PS2, reduce_time_resolution

# Load and preprocess data
# TODO: add logger
print('Loading and preprocessing data')

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
FS1 = pd.read_csv('data/FS1.txt', sep='\t', names=[str(i) + '_FS1' for i in range(1, 601)], dtype=float)
PS2 = pd.read_csv('data/PS2.txt', sep='\t', names=[str(i) + '_PS2' for i in range(1, 6001)], dtype=float)
X = pd.concat([FS1, PS2], axis=1)

# Data preprocessing + model pipeline
NCOLS_PS2 = 6000    # number of columns in PS2
STRIDE = 500        # coef of time resolution reduction

# Pipeline
pipe = make_pipeline(
    extract_PS2(ncols=NCOLS_PS2), 
    reduce_time_resolution(stride=STRIDE), 
    LogisticRegression(random_state=0, max_iter=300))

# Train and evaluate LR model
print('Training model')
# TODO: CV hyperparameter tuning
scores = cross_validate(
    pipe, 
    X, 
    y, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1), 
    scoring='accuracy', 
    return_train_score=True
    )

scores = pd.DataFrame(scores)[['train_score', 'test_score']]

# Raise error if the performance of the model is below of the threshold
class ModelPerformanceError(Exception):
    def __init__(self, metric, achieved_value, perf_threshold):
        super().__init__(f"Model performance on '{metric}' ({achieved_value}) fell below the threshold ({perf_threshold})")

perf_threshold = 0.95
if (scores.mean() < perf_threshold).any():
    raise ModelPerformanceError('accuracy', scores.mean().min(), perf_threshold)

# Train the model
clf = pipe.fit(X, y)

# Save trained model
print('Saving model')
# TODO: replace joblib by skops (https://scikit-learn.org/stable/model_persistence.html#a-more-secure-format-skops)
joblib.dump(clf, 'model.joblib')

# Save preprocessed data
print('Saving preprocessed data')
y.to_csv('data/y.csv', index=False)
X.to_csv('data/X.csv', index=False)

print('Done')
