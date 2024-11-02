import streamlit as st
from transformers import pipeline
st.title("BERT For Twitter Sentiment Analysis (MultiClass Classification)")

input = st.text_input("Enter your text here!")

classifier = pipeline(task = "text-classification",model = 'bert-base-sentiment-analysis')

if input and st.button("Evaluate"):
    result = classifier(input)
    st.write(result[0]['label'])

