import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

st.set_page_config(page_title="NLP Transformer System", page_icon="🤖")

st.title("🤖 NLP Transformer System")
st.write("Text Understanding using Transformer Models")

task = st.sidebar.selectbox("Select NLP Task", ["Sentence Similarity", "Sentiment Analysis"])

if task == "Sentence Similarity":
st.header("🔍 Sentence Similarity")

```
sentence1 = st.text_area("Sentence 1")
sentence2 = st.text_area("Sentence 2")

if st.button("Calculate Similarity"):
    if not sentence1.strip() or not sentence2.strip():
        st.warning("Please enter both sentences.")
    else:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        embedding1 = model.encode(sentence1, convert_to_tensor=True)
        embedding2 = model.encode(sentence2, convert_to_tensor=True)

        similarity = util.cos_sim(embedding1, embedding2).item()
        percentage = similarity * 100

        if similarity >= 0.70:
            result = "Similar"
        else:
            result = "Not Similar"

        st.subheader("Similarity Result")
        st.write(f"Similarity Score: {similarity:.4f}")
        st.write(f"Similarity Percentage: {percentage:.2f}%")
        st.write(f"Result: {result}")
```

elif task == "Sentiment Analysis":
st.header("😊 Sentiment Analysis")

```
text = st.text_area("Enter Text")

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )

        result = sentiment_model(text)[0]

        st.subheader("Sentiment Result")
        st.write(f"Sentiment: {result['label']}")
        st.write(f"Confidence Score: {result['score'] * 100:.2f}%")

