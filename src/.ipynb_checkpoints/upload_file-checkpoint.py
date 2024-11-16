import pandas as pd
import joblib
import numpy as np


def file_read(file):
    file = pd.read_csv(file)
    file = file.drop(columns=['Churn', 'Phone', 'Month_to_end_contract', 'Avg_class_frequency_current_month'])
    return file



    
# file = file_read(file)
# answer = []
# for i in range(50):
#     answer.append(machine_learning_best_xgb(file.iloc[[i]]))