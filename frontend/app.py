import streamlit as st
import requests

st.title(" House Price Prediction")

st.write("Enter house details")

bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0.0)
sqft_living = st.number_input("Sqft Living", min_value=0)
sqft_lot = st.number_input("Sqft Lot", min_value=0)
floors = st.number_input("Floors", min_value=0.0)

waterfront = st.number_input("Waterfront (0/1)", min_value=0)
view = st.number_input("View", min_value=0)
condition = st.number_input("Condition", min_value=0)
grade = st.number_input("Grade", min_value=0)

sqft_above = st.number_input("Sqft Above", min_value=0)
sqft_basement = st.number_input("Sqft Basement", min_value=0)

yr_built = st.number_input("Year Built", min_value=1900)
yr_renovated = st.number_input("Year Renovated", min_value=0)

zipcode = st.number_input("Zipcode", min_value=0)
lat = st.number_input("Latitude")
long = st.number_input("Longitude")

sqft_living15 = st.number_input("Sqft Living15", min_value=0)
sqft_lot15 = st.number_input("Sqft Lot15", min_value=0)

age_of_house = st.number_input("Age of House", min_value=0)
renovated = st.number_input("Renovated (0/1)", min_value=0)

if st.button("Predict Price"):

    data = {
        "bedrooms": int(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft_living": int(sqft_living),
        "sqft_lot": int(sqft_lot),
        "floors": float(floors),
        "waterfront": int(waterfront),
        "view": int(view),
        "condition": int(condition),
        "grade": int(grade),
        "sqft_above": int(sqft_above),
        "sqft_basement": int(sqft_basement),
        "yr_built": int(yr_built),
        "yr_renovated": int(yr_renovated),
        "zipcode": int(zipcode),
        "lat": float(lat),
        "long": float(long),
        "sqft_living15": int(sqft_living15),
        "sqft_lot15": int(sqft_lot15),
        "age_of_house": int(age_of_house),
        "renovated": int(renovated)
    }

    url = "http://127.0.0.1:8000/predict"

    try:

        response = requests.post(url, json=data)

        if response.status_code == 200:

            result = response.json()

            price = result["prediction"]

            st.success(f"Predicted House Price: {price:,.2f}")

        else:
            st.error("API Error")

    except:
        st.error("FastAPI server not running")

  