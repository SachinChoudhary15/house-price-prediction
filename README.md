
# House Price Prediction  

## Project Overview

This project predicts house prices using Machine Learning models like Decision Tree and Random Forest.
It provides an API interface to get predictions based on user input features.

---

## Features

* Predict house prices based on input data
* Built using FastAPI for backend
* Machine Learning models: Decision Tree & Random Forest
* Easy to deploy on cloud platforms

---

## Tech Stack

* Python
* FastAPI
* Scikit-learn
* Pandas, NumPy
* Uvicorn

---

## Project Structure

├── api/
│   └── main.py
├── model/
│   └── model.pkl
├── requirements.txt
├── app.py (optional Streamlit UI)
└── README.md

---

## Installation & Setup

### Clone the repository

clone the repository
cd house-price-prediction

### Create virtual environment

python -m venv HousePrice

### Activate environment

HousePrice\Scripts\activate

### Install dependencies

pip install -r requirements.txt

---

## Run the Project  

Run FastAPI server
uvicorn api.main:app --reload

[Open API Docs](http://127.0.0.1:8000/docs)

---

## Model Details

* Decision Tree Regressor
* Random Forest Regressor
* Trained on housing dataset

---

## Deployment

You can deploy this project using:

* Streamlit Cloud (for UI)
* Render (for FastAPI backend)

---

## Author

Sachin Choudhary
