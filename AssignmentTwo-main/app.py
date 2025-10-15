import streamlit as st
import joblib
import numpy as np

@st.cache_data
def load_model():
    return joblib.load("model.pkl")

model = load_model()
st.title("Movie Analyzler")
st.write("welcome movie person, paste a review below to have it reviewed")


user_input = st.text_area("Enter a movie review here")

if st.button("Analyze"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        prob_positive = np.round(probabilities[1] * 100, 2)
        prob_negative = np.round(probabilities[0] * 100, 2)

        if prediction == "positive":
            st.subheader(f"Predicted Sentiment: Positive")
            st.write(f"Confidence: {prob_positive}% positive | {prob_negative}% negative")
        else:
            st.subheader(f"Predicted Sentiment: Negative")
            st.write(f"Confidence: {prob_negative}% negative | {prob_positive}% positive")

    else:
        st.warning("Please enter some text before analyzing")
