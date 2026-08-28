import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

st.set_page_config(page_title="NLP Transformer System", page_icon="🤖")

st.title("🤖 NLP Transformer System")

task = st.selectbox(
"Select NLP Task",
["Sentence Similarity", "Sentiment Analysis", "Text Summarization"]
)

if task == "Sentence Similarity":
st.header("🔍 Sentence Similarity")
st.write("Compare the meaning of two sentences.")

```
sentence1 = st.text_input("Sentence 1")
sentence2 = st.text_input("Sentence 2")

button = st.button("Calculate Similarity")

if button and sentence1 and sentence2:
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)

    result = util.cos_sim(embedding1, embedding2).item()

    st.write("Similarity Score:", round(result, 4))
    st.write("Similarity Percentage:", round(result * 100, 2), "%")

    if result >= 0.70:
        st.success("✅ Similar")
    else:
        st.error("❌ Not Similar")

elif button:
    st.warning("Please enter both sentences.")
```

elif task == "Sentiment Analysis":
st.header("😊 Sentiment Analysis")
st.write("Analyze whether the text is positive or negative.")

```
text = st.text_area("Enter your text")

button = st.button("Analyze Sentiment")

if button and text:
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

    result = sentiment_model(text)[0]

    label = result["label"]
    confidence = result["score"] * 100

    if label == "POSITIVE":
        st.success("😊 Positive")
    else:
        st.error("😞 Negative")

    st.write("Confidence:", round(confidence, 2), "%")

elif button:
    st.warning("Please enter some text.")
```

else:
st.header("📝 Text Summarization")
st.write("Generate a short summary of the given text.")

```
text = st.text_area("Enter text to summarize", height=200)

button = st.button("Generate Summary")

if button and text:
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    if len(text.split()) < 30:
        st.warning("Please enter a longer paragraph.")
    else:
        summary = summarizer(
            text,
            max_length=80,
            min_length=20,
            do_sample=False
        )

        st.subheader("📄 Summary")
        st.write(summary[0]["summary_text"])

elif button:
    st.warning("Please enter some text.")

