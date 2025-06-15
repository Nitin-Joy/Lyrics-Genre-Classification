import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
model_path = "genre-bert-model"  # Folder containing saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Genre labels (ensure this matches your training classes order)
labels = [
    "Country", "Electronic", "Folk", "Hip-Hop", "Indie",
    "Jazz", "Metal", "Pop", "R&B", "Rock"
]

# Streamlit UI
st.title("🎵 Lyrics Genre Classifier")
st.write("Paste some lyrics below to predict the genre:")

lyrics_input = st.text_area("Enter lyrics here", height=200)

if st.button("Predict Genre"):
    if lyrics_input.strip() == "":
        st.warning("Please enter some lyrics.")
    else:
        # Tokenize and predict
        inputs = tokenizer(lyrics_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_index = torch.argmax(probs, dim=1).item()
            predicted_genre = labels[predicted_index]
            confidence = probs[0][predicted_index].item()

        st.success(f"🎤 Predicted Genre: **{predicted_genre}**")
        st.info(f"Confidence: **{confidence:.2%}**")

        # Optional: Show probability for all genres
        st.subheader("🔍 Genre Probabilities")
        prob_dict = {label: float(probs[0][i]) for i, label in enumerate(labels)}
        st.bar_chart(prob_dict)
