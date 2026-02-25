import gradio as gr
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# =============================================================================
# Configuration
# =============================================================================
GGUF_REPO = "antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO-GGUF"
GGUF_FILE = "Llama-3.2-3B-Instruct.Q4_K_M.gguf"  
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"

# =============================================================================
# Load RAG data
# =============================================================================
print("📚 Loading RAG data...")
with open(CHUNKS_FILE, "r") as f:
    chunks = json.load(f)
chunk_embeddings = np.load(EMBEDDINGS_FILE)
print(f"   {len(chunks)} chunks, embeddings shape: {chunk_embeddings.shape}")

# =============================================================================
# Load embedding model (CPU, fast)
# =============================================================================
print("🧮 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("   all-MiniLM-L6-v2 ready")

# =============================================================================
# Load GGUF model (CPU)
# =============================================================================
print("🤖 Downloading GGUF model...")
model_path = hf_hub_download(
    repo_id=GGUF_REPO,
    filename=GGUF_FILE,
)
print(f"   Downloaded to {model_path}")

print("🤖 Loading LLM (this takes ~30s)...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,  # CPU only
    verbose=False,
)
print("   LLM ready!")


def generate_text(prompt, max_tokens=256, temperature=0.7):
    """Generate text using GGUF model."""
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["Student Question:", "\n\nQuestion:", "###"],
    )
    return output["choices"][0]["text"].strip()


# =============================================================================
# RAG Pipeline Steps
# =============================================================================

def paraphrase_query(query):
    """Generate 2 paraphrases of the query."""
    styles = [
        ("formal and academic", "a precise, scholarly manner"),
        ("simple and inquisitive", "simple, curious language"),
    ]
    paraphrases = []
    for style_name, style_desc in styles:
        prompt = f"""Paraphrase the following question in {style_desc}.
Return ONLY the paraphrased question, nothing else.

Original: {query}

Paraphrased:"""
        result = generate_text(prompt, max_tokens=80, temperature=0.7)
        clean = result.split("\n")[0].strip().strip('"').strip("'")
        if clean and clean.lower() != query.lower():
            paraphrases.append(clean)
    return paraphrases


def retrieve_chunks(queries, top_k=5, threshold=0.3, final_top=3):
    """Retrieve relevant chunks for all query variants."""
    all_results = {}

    for q in queries:
        q_embedding = embed_model.encode([q])
        sims = cosine_similarity(q_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]

        for idx in top_indices:
            score = float(sims[idx])
            if score >= threshold:
                if idx not in all_results or score > all_results[idx]:
                    all_results[idx] = score

    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    top_results = sorted_results[:final_top]

    retrieved = []
    for idx, score in top_results:
        retrieved.append({"text": chunks[idx], "score": score, "index": idx})
    return retrieved


def generate_answer(query, context_chunks):
    """Generate Gordon Ramsay-style answer from retrieved context."""
    context = "\n\n".join([c["text"] for c in context_chunks])
    prompt = f"""You are Gordon Ramsay, but instead of cooking, you teach Deep Learning.
Answer the student's question using ONLY the provided context.
Rules:
- Be concise (max 3-4 sentences)
- Use cooking metaphors
- Be brutally honest in Gordon Ramsay's style
- Explain the concept correctly based on the context
- Do NOT use emojis

Context:
{context}

Student Question: {query}

Gordon Ramsay:"""
    return generate_text(prompt, max_tokens=200, temperature=0.7)


# =============================================================================
# Main pipeline
# =============================================================================

def rag_pipeline(query):
    """Full RAG pipeline with step-by-step outputs."""
    if not query or not query.strip():
        return "Please enter a question.", "", ""

    query = query.strip()

    # Step 1: Paraphrase
    yield "⏳ Generating paraphrases...", "", ""

    paraphrases = paraphrase_query(query)
    all_queries = [query] + paraphrases

    paraphrase_text = f"**Original:** {query}\n\n"
    for i, p in enumerate(paraphrases, 1):
        paraphrase_text += f"**Paraphrase {i}:** {p}\n\n"

    # Step 2: Retrieve
    yield paraphrase_text, "⏳ Retrieving relevant chunks...", ""

    retrieved = retrieve_chunks(all_queries)

    if not retrieved:
        yield (
            paraphrase_text,
            "⚠️ No chunks found above the 0.3 similarity threshold.",
            "I couldn't find relevant context to answer your question.",
        )
        return

    retrieval_text = ""
    for i, chunk in enumerate(retrieved, 1):
        preview = chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"]
        retrieval_text += f"**Chunk {i}** (similarity: {chunk['score']:.3f})\n\n"
        retrieval_text += f"```\n{preview}\n```\n\n"

    # Step 3: Generate answer
    yield paraphrase_text, retrieval_text, "⏳ Gordon Ramsay is thinking..."

    answer = generate_answer(query, retrieved)
    answer_display = f"🔥 **Gordon Ramsay says:**\n\n*{answer}*"

    yield paraphrase_text, retrieval_text, answer_display


# =============================================================================
# Gradio UI
# =============================================================================

EXAMPLES = [
    "What is dropout and why do we use it?",
    "Explain backpropagation.",
    "What is the vanishing gradient problem?",
    "Why do transformers use attention?",
    "What is batch normalization?",
    "Why do we use ReLU instead of sigmoid?",
]

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Gordon Ramsay RAG",
    css="footer {visibility: hidden}",
) as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 0.5em;">
            <h1>👨‍🍳 Gordon Ramsay RAG — Deep Learning Tutor</h1>
            <p style="color: #666; font-size: 1.1em;">
                Ask a Deep Learning question. Get a textbook-grounded, Ramsay-style answer.
            </p>
            <p style="color: #999; font-size: 0.9em;">
                ⚡ Running on CPU — responses take 20-40s. Be patient, unlike Ramsay.
            </p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Your Deep Learning Question",
                placeholder="e.g., What is dropout and why do we use it?",
                lines=2,
            )
        with gr.Column(scale=1, min_width=120):
            submit_btn = gr.Button("🔥 Ask Ramsay!", variant="primary", size="lg")

    gr.Examples(examples=EXAMPLES, inputs=query_input, label="Try these:")

    gr.HTML("<hr>")

    with gr.Accordion("📝 Step 1: Query Paraphrasing", open=True):
        paraphrase_output = gr.Markdown()

    with gr.Accordion("🔍 Step 2: Chunk Retrieval", open=True):
        retrieval_output = gr.Markdown()

    with gr.Accordion("🔥 Step 3: Gordon Ramsay's Answer", open=True):
        answer_output = gr.Markdown()

    gr.HTML("""
        <div style="text-align: center; padding: 1em; color: #888; font-size: 0.85em;">
            <p>
                <b>Model:</b> <a href="https://huggingface.co/antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO">Llama-3.2-3B-Gordon-Ramsay-DPO</a> |
                <b>Dataset:</b> <a href="https://huggingface.co/datasets/antonisbast/gordon-ramsay-dl-instruct">gordon-ramsay-dl-instruct</a> |
                <b>Knowledge Base:</b> 807 chunks from Introduction to Deep Learning (Notre Dame, 2025)
            </p>
            <p>Built for MSc AI & Deep Learning (AIDL_B_CS01) — University of West Attica</p>
        </div>
    """)

    submit_btn.click(
        fn=rag_pipeline,
        inputs=[query_input],
        outputs=[paraphrase_output, retrieval_output, answer_output],
    )
    query_input.submit(
        fn=rag_pipeline,
        inputs=[query_input],
        outputs=[paraphrase_output, retrieval_output, answer_output],
    )

if __name__ == "__main__":
    demo.launch()
