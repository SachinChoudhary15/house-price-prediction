


import pandas as pd 
import numpy as np 
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

def load_data(file_path):
    return pd.read_csv(file_path)

def evaluate_model(model_path, test_data_path):

    # Load trained model
    model = joblib.load(model_path)

    # Load test data
    df_test = load_data(test_data_path)
    x_test = df_test.drop("price", axis=1)
    y_test = np.log1p(df_test["price"])

    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"RMSE: {rmse}\n")
    print(f"R²: {r2}\n")
    print(f"MAE: {mae}\n")

if __name__ == "__main__":
    model_path = "models/best_model.pkl"
    test_data_path = "data/processed/final_data.csv"
    evaluate_model(model_path, test_data_path)
