# Importing the Dependencies
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Set up Streamlit app configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Sidebar for navigation
st.sidebar.title("Diabetes Prediction App")
st.sidebar.markdown("Use the form below to enter your health information and predict the chances of diabetes.")

# Data preprocessing and training the model
# Separating features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
X = standardized_data

# Train, Test, Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Support Vector Machine Classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy Score (Optional: for display)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.sidebar.text(f"Training Accuracy: {training_data_accuracy * 100:.2f}%")
st.sidebar.text(f"Testing Accuracy: {test_data_accuracy * 100:.2f}%")

# Main section of the app
st.title("Diabetes Prediction System 🩺")

st.subheader("Enter your health details:")

# Create a form to collect input data
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=25)

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# When the user submits the form
if submit_button:
    # Make prediction
    input_data = np.array(
        [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])

    # Reshape and standardize the input data
    input_data_reshaped = input_data.reshape(1, -1)
    std_data = scalar.transform(input_data_reshaped)

    # Predict diabetes
    prediction = classifier.predict(std_data)

    if prediction[0] == 0:
        st.success("The person is **not diabetic**.")
    else:
        st.error("The person is **diabetic**.")

# Additional section
st.sidebar.markdown("---")
