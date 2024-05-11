# REST API to serve ML model

import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel
import joblib
from fastapi import FastAPI

app = FastAPI()

# Load ML model
model = joblib.load("model.joblib")

# Load data
y = pd.read_csv('../data/y.csv').iloc[:, 0]
X = pd.read_csv('../data/X.csv')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_on_idx")
def predict_on_idx(idx: int):
    """Predict the target (valve condition) given the index of an example in the dataset.

    Args:
        idx (int): index of the example in the dataset.

    Returns:
        _type_: predicted valve conditon.
    """    
    prediction = model.predict(X.iloc[[idx], :])
    return prediction.item()

class DataVector(BaseModel):
    vector: list[float]

@app.post("/predict_on_x")
def predict_on_x(X: DataVector = Body(...)):
    """Predict the target (valve condition) given the feature vector.

    Args:
        X (list): list of feature values.

    Returns:
        _type_: predicted valve conditon.
    """
    df = pd.DataFrame(X.vector).T
    y_hat = model.predict(df).item()
    return {'y_hat': y_hat}
