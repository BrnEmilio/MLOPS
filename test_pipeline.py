import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

def test_model_files_exist():
    assert os.path.exists("./model.joblib"), "Modelo não encontrado"
    assert os.path.exists("./vectorizer.joblib"), "Vectorizer não encontrado"

def test_vectorizer_output_shape():
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["Este é um ótimo produto!"]
    vetor = vectorizer.transform(sample)
    assert vetor.shape[0] == 1, "Vectorizer retornou forma incorreta"
