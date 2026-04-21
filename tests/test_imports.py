def test_standard_library_imports():
    #Yes i know lol, stupid tests but i need em bro
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
    from api.app import app
    from src.inference.predict import predict_emotion
    from src.utils.mlflow_config import setup_mlflow
    from src.models.model import build_light_model
    # THIS LINE BELOW DOESNT WORK FOR TESTING PURPOSES, BUT THE FUNCTION IS DEFINITELY THERE
    from src.models.train import train_model
    from src.models.evaluate import evaluate_model
    from src.data.preprocess import preprocess_and_save

    assert app is not None
    assert callable(predict_emotion)
    assert callable(setup_mlflow)
    assert callable(build_light_model)
    # THIS LIEN BELOW DOESNT WORK FOR TESTING PURPOSES, BUT THE FUNCTION IS DEFINITELY THERE
    assert callable(train_model)
    assert callable(evaluate_model)
    assert callable(preprocess_and_save)
