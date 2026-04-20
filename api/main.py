import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


model = joblib.load("models/best_model.pkl")


class Data(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    age_of_house: int
    renovated: int


@app.get("/")
def check_api():
    return {"message": "API Running Successfully"}


@app.post("/predict")
def predict(data: Data):

    try:

        input_df = pd.DataFrame([data.dict()])

        prediction = np.expm1(model.predict(input_df))[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

