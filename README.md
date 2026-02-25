---
title: Gordon Ramsay RAG - Deep Learning Tutor
emoji: 👨‍🍳
colorFrom: red
colorTo: yellow
sdk: docker
pinned: true
license: mit
models:
  - antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO-GGUF
datasets:
  - antonisbast/gordon-ramsay-dl-instruct
---

# 🍳 Gordon Ramsay RAG - Deep Learning Tutor

An AI-powered Deep Learning tutor that answers questions in Gordon Ramsay's signature style, using Retrieval-Augmented Generation (RAG) with a DPO fine-tuned LLM.

## How It Works

1. **Document Processing**: 383-page Deep Learning textbook split into 807 chunks
2. **Query Paraphrasing**: Your question is paraphrased 2 times using the DPO model for better retrieval
3. **Semantic Retrieval**: Top chunks retrieved via cosine similarity (threshold 0.3)
4. **Answer Generation**: DPO fine-tuned Llama 3.2 3B generates a Gordon Ramsay-style answer

## Technical Stack

- **Model**: [Llama 3.2 3B Gordon Ramsay DPO](https://huggingface.co/antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO) (GGUF Q4_K_M)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)
- **Inference**: llama-cpp-python (CPU)
- **Dataset**: [gordon-ramsay-dl-instruct](https://huggingface.co/datasets/antonisbast/gordon-ramsay-dl-instruct)

## Course Project

Built for MSc AI & Deep Learning (AIDL_B_CS01) - Tasks 3 & 5 combined.
