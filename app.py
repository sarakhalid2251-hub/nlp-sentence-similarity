import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

st.set_page_config(
    page_title="NLP Transformer System",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NLP Transformer System")
st.write("A multi-task NLP application using Transformer models.")

task = st.selectbox(
    "Select NLP Task",
    [
        "Sentence Similarity",
        "Sentiment Analysis",
        "Text Summarization",
        "Named Entity Recognition",
        "Question Answering",
        "Text Generation"
    ]
)


@st.cache_resource
def load_similarity_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )


@st.cache_resource
def load_summarization_model():
    return pipeline(
        "summarization",
        model="t5-small"
    )


@st.cache_resource
def load_ner_model():
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )


@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="distilbert/distilbert-base-cased-distilled-squad"
    )


@st.cache_resource
def load_generation_model():
    return pipeline(
        "text-generation",
        model="gpt2"
    )


# --------------------------------------------------
# 1. SENTENCE SIMILARITY
# --------------------------------------------------

if task == "Sentence Similarity":

    st.header("🔍 Sentence Similarity")
    st.write("Compare the semantic meaning of two sentences.")

    sentence1 = st.text_input(
        "Sentence 1",
        placeholder="Enter the first sentence..."
    )

    sentence2 = st.text_input(
        "Sentence 2",
        placeholder="Enter the second sentence..."
    )

    if st.button("Calculate Similarity"):

        if sentence1.strip() and sentence2.strip():

            with st.spinner("Calculating similarity..."):

                model = load_similarity_model()

                embedding1 = model.encode(
                    sentence1,
                    convert_to_tensor=True
                )

                embedding2 = model.encode(
                    sentence2,
                    convert_to_tensor=True
                )

                score = util.cos_sim(
                    embedding1,
                    embedding2
                ).item()

            percentage = score * 100

            st.subheader("Result")

            st.write(
                "Similarity Score:",
                round(score, 4)
            )

            st.write(
                "Similarity Percentage:",
                round(percentage, 2),
                "%"
            )

            if score >= 0.70:
                st.success("✅ Similar")
            else:
                st.error("❌ Not Similar")

        else:
            st.warning("Please enter both sentences.")


# --------------------------------------------------
# 2. SENTIMENT ANALYSIS
# --------------------------------------------------

elif task == "Sentiment Analysis":

    st.header("😊 Sentiment Analysis")
    st.write("Analyze whether the text is positive or negative.")

    text = st.text_area(
        "Enter text",
        placeholder="Example: I really enjoyed this project."
    )

    if st.button("Analyze Sentiment"):

        if text.strip():

            with st.spinner("Analyzing sentiment..."):

                model = load_sentiment_model()
                result = model(text)[0]

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

        else:
            st.warning("Please enter some text.")


# --------------------------------------------------
# 3. TEXT SUMMARIZATION
# --------------------------------------------------

elif task == "Text Summarization":

    st.header("📝 Text Summarization")
    st.write("Generate a short summary of the given text.")

    text = st.text_area(
        "Enter text to summarize",
        height=200,
        placeholder="Enter a paragraph or longer text..."
    )

    if st.button("Generate Summary"):

        if text.strip():

            if len(text.split()) < 20:
                st.warning(
                    "Please enter a longer text for better summarization."
                )
            else:

                with st.spinner("Generating summary..."):

                    model = load_summarization_model()

                    result = model(
                        text,
                        max_length=100,
                        min_length=20,
                        do_sample=False
                    )

                st.subheader("Summary")
                st.write(result[0]["summary_text"])

        else:
            st.warning("Please enter some text.")


# --------------------------------------------------
# 4. NAMED ENTITY RECOGNITION
# --------------------------------------------------

elif task == "Named Entity Recognition":

    st.header("🏷️ Named Entity Recognition")
    st.write(
        "Identify people, organizations, locations and other entities."
    )

    text = st.text_area(
        "Enter text",
        placeholder="Example: Elon Musk works at SpaceX in the United States."
    )

    if st.button("Find Entities"):

        if text.strip():

            with st.spinner("Finding entities..."):

                model = load_ner_model()
                results = model(text)

            if results:

                st.subheader("Detected Entities")

                for entity in results:

                    st.write(
                        f"**{entity['word']}** → "
                        f"{entity['entity_group']} "
                        f"({entity['score']:.2%})"
                    )

            else:
                st.info("No named entities detected.")

        else:
            st.warning("Please enter some text.")


# --------------------------------------------------
# 5. QUESTION ANSWERING
# --------------------------------------------------

elif task == "Question Answering":

    st.header("❓ Question Answering")
    st.write("Ask a question based on the provided context.")

    context = st.text_area(
        "Context",
        height=180,
        placeholder="Enter the information here..."
    )

    question = st.text_input(
        "Question",
        placeholder="Ask a question about the context..."
    )

    if st.button("Get Answer"):

        if context.strip() and question.strip():

            with st.spinner("Finding answer..."):

                model = load_qa_model()

                result = model(
                    question=question,
                    context=context
                )

            st.subheader("Answer")
            st.success(result["answer"])

            st.write(
                "Confidence:",
                round(result["score"] * 100, 2),
                "%"
            )

        else:
            st.warning(
                "Please enter both context and question."
            )


# --------------------------------------------------
# 6. TEXT GENERATION
# --------------------------------------------------

elif task == "Text Generation":

    st.header("✍️ Text Generation")
    st.write("Generate text using a GPT-based Transformer model.")

    prompt = st.text_area(
        "Enter your prompt",
        placeholder="Example: Artificial intelligence is changing the world because..."
    )

    max_length = st.slider(
        "Maximum generated length",
        min_value=30,
        max_value=150,
        value=80
    )

    if st.button("Generate Text"):

        if prompt.strip():

            with st.spinner("Generating text..."):

                model = load_generation_model()

                result = model(
                    prompt,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )

            st.subheader("Generated Text")
            st.write(result[0]["generated_text"])

        else:
            st.warning("Please enter a prompt.")


st.divider()
st.caption(
    "Built with Python, Streamlit, PyTorch, Hugging Face Transformers "
    "and Sentence Transformers."
)
