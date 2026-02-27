# 🍳 Gordon Ramsay RAG - Deep Learning Tutor

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/antonisbast/gordon-ramsay-rag)
[![Model](https://img.shields.io/badge/🤖%20Model-Llama--3.2--3B-blue)](https://huggingface.co/antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO)
[![Dataset](https://img.shields.io/badge/📚%20Dataset-DL--Instruct-green)](https://huggingface.co/datasets/antonisbast/gordon-ramsay-dl-instruct)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

An AI-powered Deep Learning tutor that answers technical questions in Gordon Ramsay's signature brutally honest style, combining Retrieval-Augmented Generation (RAG) with Direct Preference Optimization (DPO) fine-tuning.

**🔥 [Try the Live Demo](https://huggingface.co/spaces/antonisbast/gordon-ramsay-rag)**

## 📋 Overview

This project demonstrates a complete RAG pipeline enhanced with a DPO-finetuned Large Language Model to teach Deep Learning concepts using cooking metaphors and Gordon Ramsay's unique teaching style.

### Key Features

- **📚 Knowledge Base**: 383-page Deep Learning textbook (Notre Dame, 2025) split into 807 semantic chunks
- **🔍 Smart Retrieval**: Query paraphrasing + cosine similarity for better context matching
- **🤖 Custom LLM**: Llama 3.2 3B fine-tuned with DPO on 500+ Gordon Ramsay-style Q&A pairs
- **⚡ CPU-Optimized**: Runs on CPU using quantized GGUF model (Q4_K_M)
- **🎨 Interactive UI**: Gradio interface with step-by-step visualization

## 🏗️ Architecture

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Query Paraphrasing      │
│ (2 paraphrases)         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Semantic Retrieval      │
│ - Encode query variants │
│ - Cosine similarity     │
│ - Top 3 chunks          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Answer Generation       │
│ (DPO Fine-tuned LLM)    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Gordon Ramsay Response  │
└─────────────────────────┘
```

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Llama 3.2 3B (DPO fine-tuned, GGUF Q4_K_M) |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Inference** | llama-cpp-python |
| **Vector Search** | NumPy + scikit-learn (cosine similarity) |
| **UI Framework** | Gradio |
| **Deployment** | Hugging Face Spaces (Docker) |

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9+
- 8GB+ RAM (for CPU inference)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gordon-ramsay-rag.git
cd gordon-ramsay-rag
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Access the interface**
   - Open your browser to `http://localhost:7860`

### Docker Setup

```bash
docker build -t gordon-ramsay-rag .
docker run -p 7860:7860 gordon-ramsay-rag
```

## 📊 Project Components

### 1. Data Processing
- **Source**: Introduction to Deep Learning textbook (383 pages)
- **Chunking**: Semantic splitting into 807 chunks
- **Embeddings**: Pre-computed using all-MiniLM-L6-v2

### 2. Model Training
- **Base Model**: Meta Llama 3.2 3B Instruct
- **Fine-tuning Method**: Direct Preference Optimization (DPO)
- **Training Data**: 500+ Q&A pairs in Gordon Ramsay style
- **Quantization**: Q4_K_M for efficient CPU inference

### 3. RAG Pipeline
- **Query Expansion**: 2 paraphrases per question
- **Retrieval**: Top-k search with similarity threshold
- **Context Integration**: Combines top 3 most relevant chunks
- **Response Generation**: Contextual answer with personality

## 📈 Performance

- **Inference Time**: ~20-40s per question (CPU)
- **Model Size**: ~2GB (quantized)
- **Memory Usage**: ~4-6GB RAM
- **Retrieval Accuracy**: Threshold-based filtering (0.3 cosine similarity)

## 🎯 Example Queries

- "Explain dropout and why we use it."
- "Explain backpropagation."
- "Explain the vanishing gradient problem."
- "Explain why transformers use attention."
- "Explain batch normalization."

## 📂 Project Structure

```
gordon-ramsay-rag/
├── app.py                 # Main application
├── chunks.json           # Pre-processed text chunks
├── embeddings.npy        # Pre-computed embeddings
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md           # Documentation
```

## 🎓 Academic Context

This project was developed for the MSc in AI & Deep Learning (AIDL_B_CS01) at the University of West Attica, combining:
- **Task 3**: RAG system implementation
- **Task 5**: LLM fine-tuning with DPO

## 🔗 Resources

- **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/antonisbast/gordon-ramsay-rag)
- **Fine-tuned Model**: [Llama-3.2-3B-Gordon-Ramsay-DPO](https://huggingface.co/antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO)
- **Training Dataset**: [gordon-ramsay-dl-instruct](https://huggingface.co/datasets/antonisbast/gordon-ramsay-dl-instruct)
- **GGUF Model**: [Quantized GGUF](https://huggingface.co/antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO-GGUF)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Antonis Bast**
- GitHub: [@antonisbast](https://github.com/antonisbast)
- Hugging Face: [@antonisbast](https://huggingface.co/antonisbast)

## 🙏 Acknowledgments

- Knowledge base from "Introduction to Deep Learning" (Notre Dame, 2025)
- Meta AI for Llama 3.2
- Hugging Face for infrastructure and tools

---

⭐ **If you find this project interesting, please consider starring it on GitHub!**
