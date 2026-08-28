import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("🤖 NLP Sentence Similarity")
st.write("Compare the meaning of two sentences.")

sentence1 = st.text_input("Sentence 1")
sentence2 = st.text_input("Sentence 2")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

button = st.button("Calculate Similarity")

result = util.cos_sim(model.encode(sentence1), model.encode(sentence2)).item() if button and sentence1 and sentence2 else None

st.write("Similarity Score:", round(result, 4) if result is not None else "Enter both sentences and click the button.")
