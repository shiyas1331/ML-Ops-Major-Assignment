# tests/test_train.py

import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def test_data_load():
    data = fetch_california_housing()
    assert data.data.shape[0] > 0, "Dataset not loaded properly"

def test_model_type():
    model = LinearRegression()
    assert isinstance(model, LinearRegression), "Model is not LinearRegression"

def test_model_training():
    data = fetch_california_housing()
    X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, "coef_"), "Model not trained properly"

def test_model_r2_score():
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    assert r2 > 0.5, f"R2 score too low: {r2}"
