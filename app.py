import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

st.set_page_config(page_title="NLP Transformer System", page_icon="🤖")

st.title("🤖 NLP Transformer System")
st.write("A simple NLP application using Transformer models.")

task = st.selectbox(
"Select NLP Task",
[
"Sentence Similarity",
"Sentiment Analysis",
"Text Summarization"
]
)

if task == "Sentence Similarity":
st.header("🔍 Sentence Similarity")

```
sentence1 = st.text_input("Sentence 1")
sentence2 = st.text_input("Sentence 2")

if st.button("Calculate Similarity"):
    if not sentence1 or not sentence2:
        st.warning("Please enter both sentences.")
    else:
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        embedding1 = model.encode(sentence1)
        embedding2 = model.encode(sentence2)

        similarity = util.cos_sim(
            embedding1,
            embedding2
        ).item()

        percentage = similarity * 100

        st.write(
            "Similarity Score:",
            round(similarity, 4)
        )

        st.write(
            "Similarity Percentage:",
            round(percentage, 2),
            "%"
        )

        if similarity >= 0.70:
            st.success("✅ Similar")
        else:
            st.error("❌ Not Similar")
```

elif task == "Sentiment Analysis":
st.header("😊 Sentiment Analysis")

```
text = st.text_area("Enter your text")

if st.button("Analyze Sentiment"):
    if not text:
        st.warning("Please enter some text.")
    else:
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )

        result = sentiment_model(text)[0]

        label = result["label"]
        confidence = result["score"] * 100

        if label == "POSITIVE":
            st.success("😊 Positive")
        else:
            st.error("😞 Negative")

        st.write(
            "Confidence:",
            round(confidence, 2),
            "%"
        )
```

else:
st.header("📝 Text Summarization")

```
text = st.text_area(
    "Enter text to summarize",
    height=200
)

if st.button("Generate Summary"):
    if not text:
        st.warning("Please enter some text.")
    elif len(text.split()) < 30:
        st.warning(
            "Please enter a longer paragraph for summarization."
        )
    else:
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1
        )

        summary = summarizer(
            text,
            max_length=80,
            min_length=20,
            do_sample=False
        )

        st.subheader("📄 Summary")
        st.write(summary[0]["summary_text"])

