import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("🤖 NLP Transformer System")
st.write("Sentence Similarity using Transformer Models")

sentence1 = st.text_input("Sentence 1")
sentence2 = st.text_input("Sentence 2")

button = st.button("Calculate Similarity")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

if button and sentence1 and sentence2:
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
similarity = util.cos_sim(embedding1, embedding2).item()
st.write("Similarity Score:", round(similarity, 4))
st.write("Similarity Percentage:", round(similarity * 100, 2), "%")
st.write("Result:", "Similar" if similarity >= 0.70 else "Not Similar")
