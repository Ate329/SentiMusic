import streamlit as st

temperature = st.slider("Temperature (Randomness)", min_value=0.00, max_value=1.00)

audio_length = st.slider("Audio Length (in second)", min_value=0)
