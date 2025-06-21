import streamlit as st
import joblib
import os

st.title("Classificador de Sentimento")

texto = st.text_input("Digite um tweet")

if os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib"):
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    if st.button("Analisar"):
        if texto.strip():
            vetor = vectorizer.transform([texto])
            pred = model.predict(vetor)[0]
            st.write(f"Sentimento: {pred}")
        else:
            st.warning("Por favor, insira um texto para análise.")
else:
    st.error("Modelo ou vetorizador não encontrado. Certifique-se de que os arquivos model.joblib e vectorizer.joblib estão presentes.")
