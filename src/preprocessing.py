

import pandas as pd
import numpy as np
import os
import joblib

# load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def preprocessed_data(input_file_path, output_file_path):

    df = load_data(input_file_path)

    # Drop null values
    df = df.dropna()


    # Remove outliers from target variable
    df = remove_outliers(df, "price")

    # Drop unuseful columns
    df = df.drop(["id", "date"], axis=1, errors="ignore")

    # Save the processed data
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"Preprocessed data saved at {output_file_path}")

if __name__ == "__main__":
    input_file_path = "data/raw/kc_house_data.csv"
    output_file_path = "data/processed/processed_data.csv"
    preprocessed_data(input_file_path, output_file_path)


