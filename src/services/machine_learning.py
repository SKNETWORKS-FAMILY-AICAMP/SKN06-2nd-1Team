import joblib
import os
def load_model_best_xgb():
    model_path = os.path.join('models', 'best_xgb.pkl')
    return joblib.load(model_path)

def machine_learning_best_xgb(input_data):
    model = load_model_best_xgb()
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1] 
    return prediction, prediction_proba
    
# def load_model_best_gb():
#     model_path = os.path.join('models', 'best_gb.pkl')
#     return joblib.load(model_path)
# def load_model_best_rf():
#     model_path = os.path.join('models', 'best_rf.pkl')
#     return joblib.load(model_path)


# 머신 러닝

# def machine_learning_best_gb(input_data):
#     model = load_model_best_gb()
#     prediction = model.predict(input_data)
#     prediction_proba = model.predict_proba(input_data)[:, 1] 
#     return prediction, prediction_proba

# def machine_learning_best_rf(input_data):
#     model = load_model_best_rf()
#     prediction = model.predict(input_data)
#     prediction_proba = model.predict_proba(input_data)[:, 1] 
#     return prediction, prediction_proba