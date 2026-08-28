
import gradio as gr
from sentence_transformers import util

def calculate_similarity(sentence1, sentence2):
    if not sentence1.strip() or not sentence2.strip():
        return "⚠️ Please enter both sentences."

    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    similarity = util.cos_sim(embedding1, embedding2).item()
    percentage = similarity * 100

    if similarity >= 0.70:
        result = "✅ Similar"
    else:
        result = "❌ Not Similar"

    return (
        f"Similarity Score: {similarity:.4f}\n"
        f"Similarity Percentage: {percentage:.2f}%\n"
        f"Result: {result}"
    )


demo = gr.Interface(
    fn=calculate_similarity,
    inputs=[
        gr.Textbox(
            label="Sentence 1",
            placeholder="Enter first sentence..."
        ),
        gr.Textbox(
            label="Sentence 2",
            placeholder="Enter second sentence..."
        )
    ],
    outputs=gr.Textbox(label="Similarity Result"),
    title="🤖 NLP Transformer - Sentence Similarity",
    description="Compare the semantic meaning of two sentences using a Transformer model.",
    examples=[
        [
            "I love programming in Python.",
            "Python programming is something I really enjoy."
        ],
        [
            "The cat is sitting on the mat.",
            "A cat is sitting on a mat."
        ],
        [
            "The weather is very hot today.",
            "It is extremely warm outside today."
        ],
        [
            "The cat is sitting on the mat.",
            "A dog is running in the park."
        ]
    ]
)

demo.launch(share=True)
