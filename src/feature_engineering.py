

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def load_data(file_path):
    return pd.read_csv(file_path)

def encode_categorical_columns(df):
    label_encoders = {}

    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def feature_engineering(input_file_path, output_file_path):
    df = load_data(input_file_path)

    # Create new features
    current_year = datetime.now().year
    df["age_of_house"] = current_year - df["yr_built"]
    df["renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["has_basement"] = (df["sqft_basement"] > 0).astype(int)


    df, encoders = encode_categorical_columns(df)
    # Save encoders
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/label_encoders.pkl")

    # Save final dataset
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"Feature engineered data saved at {output_file_path}")

if __name__ == "__main__":
    input_file_path = "data/processed/processed_data.csv"
    output_file_path = "data/processed/final_data.csv"
    feature_engineering(input_file_path, output_file_path)