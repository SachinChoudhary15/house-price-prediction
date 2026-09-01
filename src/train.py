


import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def load_data(file_path):
    return pd.read_csv(file_path)

def train_model(input_file_path, model_output_path):
    df = load_data(input_file_path)
    x = df.drop("price", axis=1)
    y = np.log1p(df["price"]) 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    

    # Decision Tree Regressor pipeline
    dtr_pipeline = Pipeline([
        ("model", DecisionTreeRegressor(random_state=42))
    ])

    # Random Forest Regressor pipeline
    rfr_pipeline = Pipeline([
        ("model", RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter tuning for Decision Tree Regressor
    param_distributions = {
    "model__max_depth":[None,10,20,30,50],
    "model__min_samples_split": [10, 20, 30, 40],
    "model__min_samples_leaf": [5, 10, 15, 20],
    "model__max_features": [None,"sqrt","log2"],
    "model__ccp_alpha": [0.0, 0.001, 0.01]
    }

    random_search_dtr = RandomizedSearchCV(
    estimator=dtr_pipeline,
    param_distributions=param_distributions,

    n_iter=25,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring="r2",
    return_train_score=True
    )



    random_search_dtr.fit(x_train, y_train)
    best_dtr = random_search_dtr.best_estimator_

    y_train_pred_dtr = best_dtr.predict(x_train)
    y_test_pred_dtr = best_dtr.predict(x_test)

    dtr_train_r2 = r2_score(y_train, y_train_pred_dtr)
    dtr_test_r2 = r2_score(y_test, y_test_pred_dtr)

    actual_price_dtr = np.expm1(y_test)
    predicted_price_dtr = np.expm1(y_test_pred_dtr)

    original_r2_dtr = r2_score(actual_price_dtr,predicted_price_dtr)

    rmse = np.sqrt(mean_squared_error(actual_price_dtr,predicted_price_dtr))

    mae = mean_absolute_error(actual_price_dtr,predicted_price_dtr)

    print(f"Original Price R2 Score DTR: {original_r2_dtr}")



    print(f"rmse: {rmse}\n")
    print(f"mae: {mae}\n")

    print(f"Overfitting_ Gap: {dtr_train_r2 - dtr_test_r2}\n")
    print(f"Best Hyperparameters: {random_search_dtr.best_params_}\n")

    # Hyperparameter tuning for Random Forest Regressor
    param_distributions_rfr = {

    "model__n_estimators": [300, 400, 500],
    "model__max_depth": [10, 15, 20],
    "model__min_samples_split": [5, 10, 15, 20],
    "model__min_samples_leaf": [2, 4, 6, 8],
    "model__max_features": ["sqrt", "log2"],
    "model__bootstrap": [True],
    "model__criterion": ["squared_error"],
}
    
    random_search_rfr = RandomizedSearchCV(
    estimator=rfr_pipeline,
    param_distributions=param_distributions_rfr,
    n_iter=20,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring="r2",
    return_train_score=True,
)

    random_search_rfr.fit(x_train, y_train)
    best_rfr = random_search_rfr.best_estimator_

    y_train_pred_rfr = best_rfr.predict(x_train)
    y_test_pred_rfr = best_rfr.predict(x_test)

    rfr_train_r2 = r2_score(y_train, y_train_pred_rfr)
    rfr_test_r2 = r2_score(y_test, y_test_pred_rfr)

    actual_price = np.expm1(y_test)
    predicted_price = np.expm1(y_test_pred_rfr)

    original_r2 = r2_score(actual_price, predicted_price)

    print(f"Original Price R2 Score: {original_r2}")

    rmse = np.sqrt(mean_squared_error(actual_price, predicted_price))
    mae = mean_absolute_error(actual_price, predicted_price)

    print(f"\nrfr_train_r2: {rfr_train_r2}\n")
    print(f"rfr_test_r2: {rfr_test_r2}\n")

    print(f"rmse: {rmse}\n")
    print(f"mae: {mae}\n")

    print(f"Overfitting_ Gap: {rfr_train_r2 - rfr_test_r2}\n")
    print(f"Best Hyperparameters: {random_search_rfr.best_params_}\n")    

    # Determine best model 
    if rfr_test_r2 > dtr_test_r2:
        best_model = best_rfr
        print("Best Model: Random Forest Regressor\n")
    else:
        best_model = best_dtr
        print("Best Model: Decision Tree Regressor\n")

    # Save the best model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"Best model saved at {model_output_path}")

if __name__ == "__main__":
    input_file_path = "data/processed/final_data.csv"
    model_output_path = "models/best_model.pkl"
    train_model(input_file_path, model_output_path)