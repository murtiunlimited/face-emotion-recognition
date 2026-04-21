def test_standard_library_imports():
    import os
    import io
    import time

    assert os is not None
    assert io is not None
    assert time is not None

def test_core_ml_libraries():
    import numpy
    import cv2

    assert numpy is not None
    assert cv2 is not None


def test_fastapi_stack():
    import fastapi
    from fastapi.testclient import TestClient

    assert fastapi is not None
    assert TestClient is not None

def test_ml_stack_imports():
    import tensorflow
    import mlflow
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from tqdm import tqdm

    assert tensorflow is not None
    assert mlflow is not None
    assert train_test_split is not None
    assert classification_report is not None
    assert accuracy_score is not None
    assert tqdm is not None

def test_project_imports():

    from src.models.train import train_model

    assert callable(train_model)
  
