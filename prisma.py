import streamlit as st
import pandas as pd
import joblib
import json
import os

# Get the current working directory of the Streamlit app
app_directory = os.path.dirname(os.path.abspath(__file__))

# Load the trained model using the relative path
model_path = os.path.join(app_directory, 'xgb_model_ov.pkl')
model = joblib.load(model_path)

# import pickle
# # Save model
# with open('model.pkl', 'wb') as file:
#     pickle.dump('xgb_model_ov.pkl', file)


# Load column mappings from JSON file using the relative path
mappings_path = os.path.join(app_directory, 'mappings.json')
with open(mappings_path, 'r') as file:
    column_mappings = json.load(file)


# Convert string keys back to integers (optional)
column_mappings = {col: {int(k): v for k, v in mapping.items()} for col, mapping in column_mappings.items()}
column_mappings['occupation_skillScore'] = {1: "Low", 2: "Medium", 3: "High"}
column_mappings["family_history"] =  {1: "Yes", 2: "No"}

# Reverse mappings for user input
reverse_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in column_mappings.items()}


# Streamlit app
# st.title("Preventable Factors-based Risk Indicator for Screening and Measurement of Alzheimer’s (PRISMA)")

# Streamlit app
st.markdown(
    """
    <style>
    .title {
        color:  #007bff;
        text-align: center;
    }
    .big-button > button {
        font-size: 40px;  /* Increase font size significantly */
        padding: 50px 100px;  /* Increase padding to make the button much bigger */
        width: 400px;  /* Set button width much larger */
        margin: 20px auto;  /* Center the button */
        display: block;  /* Ensure the button is a block element */
        border-radius: 12px;  /* Optional: Add rounded corners */
        background-color: #007bff;  /* Optional: Change button color */
    }
    .stButton > button {
        color: white;  /* Change text color */
        background-color: #007bff;  /* Button background color */
    }
    body {
        background-color: white;  /* Set background color to white */
    }
    .user-input-table {
        width: 100%;  /* Set width to 100% of the parent container */
        max-width: 100%;  /* Ensure it doesn't exceed 100% of the container width */
        margin: 0 auto;  /* Center align the table */
        display: block;
    }
    .stDataFrame {
        width: 100% !important;  /* Ensure DataFrame takes full width */
    }
    </style>
    """, unsafe_allow_html=True)



st.markdown('<h1 class="title">Preventable Factors-based Risk Indicator for Screening and Measurement of Alzheimer’s (PRISMA)</h1>', unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.header("User Input Features")
user_input = {}

for col, mapping in column_mappings.items():
    user_input[col] = st.sidebar.selectbox(
        f"Select {col}",
        options=list(mapping.values()),
        index=0
    )

# Convert user input to numerical values
numerical_input = {col: reverse_mappings[col][value] for col, value in user_input.items()}

# Convert input to DataFrame for prediction
input_df = pd.DataFrame([numerical_input])

# Ensure correct column order for the model
input_df = input_df[['gender', 'education', 'spouse_edu', 'category', 'religion', 
                     'occupation_skillScore', 'marital_status', 'age_bin', 
                     'family_history', 'hhold_asset_bin']]

# Display the user input DataFrame with the added custom CSS class
st.write("### User Input:", unsafe_allow_html=True)
st.write(input_df.style.set_table_attributes('class="user-input-table"'))

# Make prediction and display probability
with st.container():
    if st.button("Predict", key="predict_button"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Convert prediction to "Yes" or "No"
        prediction_label = "Yes" if prediction == 1 else "No"
        
        # Display prediction and probability
        st.write("### Prediction:")
        st.write(f"The model predicts: **{prediction_label}**")
        st.write(f"### Probability:")
        st.write(f"The probability of the prediction being 'Yes' is: **{probability:.4f}**")
