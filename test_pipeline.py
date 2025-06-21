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

def test_model_prediction_labels():
    model = joblib.load("./model.joblib")
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["O serviço foi péssimo"]
    vetor = vectorizer.transform(sample)
    pred = model.predict(vetor)[0]
    assert pred in ["positivo", "negativo"], f"Rótulo inesperado: {pred}"

def test_data_validation():
    df = pd.read_csv("./data/tweets_limpo.csv")
    assert "text" in df.columns and "label" in df.columns
    assert df["text"].notnull().all()
    assert df["label"].isin(["positivo", "negativo"]).all()

def test_fairness_by_text_length():    
    # Esse teste vai garantir que o modelo tenha um desempenho semelhante (em acurácia) para textos curtos, médios e longos. (Se a diferença passar de 20%, ele falha)
    model_path = "./model.joblib"
    vectorizer_path = "./vectorizer.joblib"
    data_path = "./data/tweets_limpo.csv"

    assert os.path.exists(model_path), "Modelo não encontrado"
    assert os.path.exists(vectorizer_path), "Vetorizador não encontrado"
    assert os.path.exists(data_path), "Arquivo de dados não encontrado"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    df = pd.read_csv(data_path)

    df["text_len"] = df["text"].apply(len)
    df["len_category"] = pd.cut(df["text_len"], bins=[0, 50, 150, 1000], labels=["curto", "medio", "longo"])
    X = vectorizer.transform(df["text"])
    y_true = df["label"]
    y_pred = model.predict(X)

    results = {}
    for cat in df["len_category"].unique():
        subset = df[df["len_category"] == cat]
        if not subset.empty:
            X_sub = vectorizer.transform(subset["text"])
            y_sub_true = subset["label"]
            y_sub_pred = model.predict(X_sub)
            acc = accuracy_score(y_sub_true, y_sub_pred)
            results[str(cat)] = acc

    # Exemplo de fairness: diferença máxima de acurácia permitida entre grupos = 0.2 (20%)
    acc_values = list(results.values())
    max_diff = max(acc_values) - min(acc_values)
    print("Acurácias por grupo de tamanho:", results)
    assert max_diff < 0.2, f"Diferença de acurácia entre grupos muito alta! ({max_diff:.2f})"