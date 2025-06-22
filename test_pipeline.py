import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

def test_model_files_exist():
    assert os.path.exists("./model.joblib"), "Modelo não encontrado"
    assert os.path.exists("./vectorizer.joblib"), "Vectorizer não encontrado"
