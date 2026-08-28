import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("🤖 NLP Transformer System")
st.write("Sentence Similarity and Text Understanding")

sentence1 = st.text_input("Sentence 1")
sentence2 = st.text_input("Sentence 2")

button = st.button("Calculate Similarity")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

result = util.cos_sim(model.encode(sentence1), model.encode(sentence2)).item() if button and sentence1 and sentence2 else None

st.write("Similarity Score:", round(result, 4) if result is not None else "Enter both sentences.")

st.write("Similarity Percentage:", round(result * 100, 2) if result is not None else "")

st.write("Result:", "✅ Similar" if result is not None and result >= 0.70 else "❌ Not Similar" if result is not None else "")
