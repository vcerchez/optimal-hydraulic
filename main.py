# REST API to serve ML model

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi import FastAPI

app = FastAPI()

# Load ML model
model = joblib.load("model.joblib")

# Load data
y = pd.read_csv('data/y.csv').iloc[:, 0]
X = pd.read_csv('data/X.csv')


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
