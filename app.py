
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="NLP Sentence Similarity",
    page_icon="🤖"
)

st.title("🤖 NLP Transformer - Sentence Similarity")
st.write(
    "Compare the semantic meaning of two sentences using a Transformer model."
)

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

model = load_model()

sentence1 = st.text_area(
    "Sentence 1",
    placeholder="Enter first sentence..."
)

sentence2 = st.text_area(
    "Sentence 2",
    placeholder="Enter second sentence..."
)

if st.button("Calculate Similarity"):

    if not sentence1.strip() or not sentence2.strip():
        st.warning("⚠️ Please enter both sentences.")

    else:
        embedding1 = model.encode(
            sentence1,
            convert_to_tensor=True
        )

        embedding2 = model.encode(
            sentence2,
            convert_to_tensor=True
        )

        similarity = util.cos_sim(
            embedding1,
            embedding2
        ).item()

        percentage = similarity * 100

        if similarity >= 0.70:
            result = "✅ Similar"
        else:
            result = "❌ Not Similar"

        st.subheader("Similarity Result")
        st.write(f"**Similarity Score:** {similarity:.4f}")
        st.write(f"**Similarity Percentage:** {percentage:.2f}%")
        st.write(f"**Result:** {result}")

