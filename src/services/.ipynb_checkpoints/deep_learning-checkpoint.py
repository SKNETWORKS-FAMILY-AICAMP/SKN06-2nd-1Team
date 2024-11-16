import joblib
import torch
import os

# 딥 러닝
def deep_learning(input_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join("models", 'dout_model.pt')
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    scaler = joblib.load(os.path.join(os.getcwd(),"models/scaler.pkl"))
    
    input_data = scaler.transform(input_data)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        churn_probability = model(input_tensor).item()

    return churn_probability

