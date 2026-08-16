# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load model from deployment folder
MODEL_PATH = Path("tourism_project/deployment")/ "tourism_model.pkl"

model = joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Tourism Package Prediction")
st.write(
    "Enter customer details to predict whether the customer "
    "will purchase the tourism package."
)


# User inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)

    typeofcontact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    citytier = st.selectbox("City Tier", [1, 2, 3])

    durationofpitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        value=10
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox("Gender", ["Male", "Female"])

    numberofpersonvisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=2
    )

    numberoffollowups = st.number_input(
        "Number of Followups",
        min_value=0,
        value=3
    )

    productpitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )


with col2:
    preferredpropertystar = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    maritalstatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    numberoftrips = st.number_input(
        "Number of Trips",
        min_value=0,
        value=2
    )

    passport = st.selectbox("Passport", [0, 1])

    pitchsatisfactionscore = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5]
    )

    owncar = st.selectbox("Own Car", [0, 1])

    numberofchildrenvisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=1
    )

    designation = st.selectbox(
        "Designation",
        ["AVP", "VP", "Manager", "Senior Manager", "Executive"]
    )

    monthlyincome = st.number_input(
        "Monthly Income",
        min_value=0,
        value=25000
    )


# Create dataframe
input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [typeofcontact],
    "CityTier": [citytier],
    "DurationOfPitch": [durationofpitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [numberofpersonvisiting],
    "NumberOfFollowups": [numberoffollowups],
    "ProductPitched": [productpitched],
    "PreferredPropertyStar": [preferredpropertystar],
    "MaritalStatus": [maritalstatus],
    "NumberOfTrips": [numberoftrips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitchsatisfactionscore],
    "OwnCar": [owncar],
    "NumberOfChildrenVisiting": [numberofchildrenvisiting],
    "Designation": [designation],
    "MonthlyIncome": [monthlyincome]
})


# Prediction
if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(
            "🎉 Customer is likely to purchase the tourism package."
        )
    else:
        st.info(
            "Customer is unlikely to purchase the tourism package."
        )

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

        st.metric(
            "Purchase Probability",
            f"{probability:.2%}"
        )

    st.subheader("Customer Input")
    st.dataframe(input_data)
