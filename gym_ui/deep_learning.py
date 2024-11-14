import streamlit as st
import joblib
import torch
import os
def load_model():
    model_path = os.path.join(os.getcwd(), 'models/best_rf.pkl')
    return joblib.load(model_path)


# 딥 러닝
def deep_learning(input_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(os.getcwd(), 'models/dout_model.pt')
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    scaler = joblib.load(os.path.join(os.getcwd(),"models/scaler.pkl"))
    
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        churn_probability = model(input_tensor).item()

    return churn_probability


# 머신 러닝
def machine_learning(input_data):
    model = load_model()
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1] 
    return prediction, prediction_proba