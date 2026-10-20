# Simple, Modular RAG (Retrieval-Augmented Generation)

A clean, production-grade, and 100% understandable implementation of Retrieval-Augmented Generation (RAG) built from scratch in Python.

No opaque abstractions (no LangChain, no LlamaIndex, no hidden vector store bloatware). Every step—from chunking to cosine similarity to prompt augmentation—is transparent, readable, and extendable.

---

## Project Structure

```text
rag/
├── data/
│   └── knowledge_base.txt      # Sample knowledge document (Quantum Facility Protocols)
├── src/
│   ├── __init__.py
│   ├── chunker.py              # Sentence-aware text chunking with sliding overlap
│   ├── embeddings.py           # Pluggable vector embedding providers (Ollama & Mock)
│   ├── vector_store.py         # Pure-math vector database with Cosine Similarity search
│   ├── llm.py                  # Generative LLM client interface (Ollama & Mock)
│   └── pipeline.py             # End-to-end RAG orchestrator
├── tests/
│   └── test_rag.py             # Automated unit tests for chunking, similarity math & store
├── main.py                     # CLI entrypoint (Demo mode, Single query, Interactive REPL)
├── Explanation.yaml            # State-by-state mechanics vs LangChain under-the-hood
├── EXPLANATION.md              # Complete 100% conceptual & mathematical deep dive
├── requirements.txt            # Minimal Python dependencies
└── README.md                   # Quickstart guide
```

---

## Requirements & Prerequisites

### 1. Python Environment
* Python 3.10 or higher
* Install minimal dependencies:
```bash
pip install -r requirements.txt
```

### 2. Local Ollama & Models
This project runs 100% locally with zero cloud or API key dependencies using [Ollama](https://ollama.com).

#### Install the Default Models:
```bash
# 1. Pull the default embedding model (384-dimensional dense semantic vectors, ~45MB)
ollama pull all-minilm

# 2. Pull the default generative LLM (3-billion parameter coding/reasoning model, ~1.9GB)
ollama pull qwen2.5-coder:3b
```

#### Start the Ollama Service:
Make sure the local Ollama background server is running:
```bash
ollama serve
```

> [!TIP]
> **Using Custom Models**: You can swap in any other models available in your Ollama library at runtime using flags:
> ```bash
> python3 main.py --embed-model nomic-embed-text --llm-model llama3.2
> ```
> *(Or run completely offline with simulated components using `python3 main.py --mock`)*

---

## Quickstart

### 1. Run with Multi-Document Directory (Default)
By default, the pipeline scans the entire `data/` directory and indexes all `.pdf`, `.txt`, and `.md` files:
```bash
python3 main.py
```

### 2. Index Any Custom Folder or File
Point `--data` to any folder or single file:
```bash
# Index an entire folder of PDFs and text files
python3 main.py --data /path/to/my_documents/ --interactive

# Or index a single file
python3 main.py --data data/erp.pdf --query "Who is the author of Chapter V?"
```

### 3. Ask Single Questions from the CLI
```bash
python3 main.py --query "What are the strategic perspectives of ERP systems?"
```

### 4. Interactive Q&A Shell (REPL)
Chat with your documents in real time:
```bash
python3 main.py --interactive
```

### 4. Run Completely Offline (Mock Mode)
Run instantly without any background services or LLMs:
```bash
python3 main.py --mock
```

### 5. Run the Test Suite
```bash
python3 -m unittest discover -s tests
```

---

