import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import os
import json
from groq import Groq

# Load the diabetes dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Preprocess the data
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)

# Train the SVM model
classifier = svm.SVC(kernel='linear')
classifier.fit(X, Y)

# Streamlit app configuration
st.set_page_config(
    page_title="Diabetes Prediction & LLAMA Chatbot",
    page_icon="🩸",
    layout="wide"
)

# Sidebar for LLAMA chatbot
with st.sidebar:
    st.title("🦙 LLAMA 3.1 Chatbot")

    # Load Groq API key from config file
    working_dir = os.path.dirname(os.path.abspath(__file__))
    config_data = json.load(open(f"{working_dir}/config(2).json"))

    GROQ_API_KEY = config_data["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

    client = Groq()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user's message
    user_prompt = st.text_input("Ask LLAMA...")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Send user's message to the LLM and get a response
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            *st.session_state.chat_history
        ]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the LLM's response
        st.markdown(assistant_response)

# Main section for Diabetes Prediction
st.title("Intelligent Diabetes Prediction System 🩺")

# Create form for user input
with st.form("diabetes_form"):
    st.subheader("Please enter your health details:")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, step=1)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# When the user submits the form
if submit_button:
    # Make prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    input_data = scalar.transform(input_data)
    prediction = classifier.predict(input_data)

    if prediction[0] == 0:
        st.success("The person is **not diabetic**.")
    else:
        st.error("The person is **diabetic**.")
