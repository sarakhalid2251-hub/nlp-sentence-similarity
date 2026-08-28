import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

st.set_page_config(page_title="NLP Transformer System", page_icon="🤖")

st.title("🤖 NLP Transformer System")
st.write("Text Understanding using Transformer Models")

task = st.sidebar.radio("Select NLP Task", ["Sentence Similarity", "Sentiment Analysis"])

sentence1 = ""
sentence2 = ""
text = ""

if task == "Sentence Similarity":
st.header("🔍 Sentence Similarity")
sentence1 = st.text_area("Sentence 1")
sentence2 = st.text_area("Sentence 2")

```
if st.button("Calculate Similarity"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2).item()
    st.write("Similarity Score:", round(similarity, 4))
    st.write("Similarity Percentage:", round(similarity * 100, 2), "%")

    if similarity >= 0.70:
        st.success("Similar")
    else:
        st.error("Not Similar")
```

else:
st.header("😊 Sentiment Analysis")
text = st.text_area("Enter Text")

```
if st.button("Analyze Sentiment"):
    sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    result = sentiment_model(text)[0]
    st.write("Sentiment:", result["label"])
    st.write("Confidence:", round(result["score"] * 100, 2), "%")

