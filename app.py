import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("🤖 NLP Sentence Similarity")
st.write("Compare the meaning of two sentences.")

sentence1 = st.text_input("Sentence 1")
sentence2 = st.text_input("Sentence 2")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

button = st.button("Calculate Similarity")

result = util.cos_sim(model.encode(sentence1), model.encode(sentence2)).item() if button and sentence1 and sentence2 else None

if result is not None:
    st.write("Similarity Score:", round(result, 4))
    st.write("Similarity Percentage:", round(result * 100, 2), "%")

    if result >= 0.70:
        st.success("✅ Similar")
    else:
        st.error("❌ Not Similar")
else:
    st.write("Enter both sentences and click the button.")
