# Gordon Ramsay RAG - Deep Learning Tutor

An AI-powered Deep Learning tutor combining RAG with a DPO-finetuned LLM that teaches in Gordon Ramsay's signature style.

🔥 **[Try the Live Demo on Hugging Face](https://huggingface.co/spaces/antonisbast/gordon-ramsay-rag)**

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/gordon-ramsay-rag.git
cd gordon-ramsay-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python app.py
```

## Features

- 📚 RAG system with 807 knowledge chunks from DL textbook
- 🤖 Custom DPO fine-tuned Llama 3.2 3B
- ⚡ CPU-optimized inference (GGUF quantization)
- 🎨 Interactive Gradio interface

See [full documentation](README_GITHUB.md) for technical details and architecture.

## Resources

- **Model**: [Llama-3.2-3B-Gordon-Ramsay-DPO](https://huggingface.co/antonisbast/Llama-3.2-3B-Gordon-Ramsay-DPO)
- **Dataset**: [gordon-ramsay-dl-instruct](https://huggingface.co/datasets/antonisbast/gordon-ramsay-dl-instruct)

## License

MIT License - Built for MSc AI & Deep Learning @ University of West Attica
