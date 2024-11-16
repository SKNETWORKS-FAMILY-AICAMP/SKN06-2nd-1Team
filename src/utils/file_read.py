import pandas as pd
import joblib


def file_read(file):
    file = pd.read_csv(file)
    file = file.drop(columns=['Churn', 'Phone', 'Month_to_end_contract', 'Avg_class_frequency_current_month'])
    return file