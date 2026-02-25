---
title: Gordon Ramsay RAG - Deep Learning Tutor
emoji: 👨‍🍳
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.6.0
python_version: '3.12'
app_file: app.py
pinned: true
license: mit
tags:
  - rag
  - dpo
  - llama
  - gordon-ramsay
  - deep-learning
  - nlp
models:
  - antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO
datasets:
  - antonisbast/gordon-ramsay-dl-instruct
---

# 👨‍🍳 Gordon Ramsay RAG — Deep Learning Tutor

Ask any Deep Learning question and get a **Gordon Ramsay-style answer** grounded in a real textbook, powered by a custom RAG pipeline.

⚡ Runs entirely on CPU using a GGUF-quantized model. Responses take 20-40 seconds.

## How It Works

1. **Paraphrase** — Your query is rephrased 2 times using the DPO-trained Llama 3.2 3B model
2. **Retrieve** — Top chunks are retrieved from a 383-page Deep Learning textbook via cosine similarity
3. **Generate** — The same model produces a Gordon Ramsay-style answer grounded in the retrieved context

## Technical Details

- **Model:** [Llama-3.2-3B-Gordon-Ramsay-DPO](https://huggingface.co/antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO) — GGUF Q4_K_M (~2GB)
- **Embeddings:** all-MiniLM-L6-v2 (384-dim, CPU)
- **Knowledge Base:** 807 chunks from *Introduction to Deep Learning* (Notre Dame, 2025)
- **Dataset:** [gordon-ramsay-dl-instruct](https://huggingface.co/datasets/antonisbast/gordon-ramsay-dl-instruct) (500 train / 100 test)
- **Inference:** llama-cpp-python on CPU (no GPU required)

Built as part of the MSc AI & Deep Learning program (AIDL_B_CS01) at the University of West Attica.