import os
import sys
import argparse
from src.chunker import TextChunker
from src.embeddings import OllamaEmbedding, MockEmbedding
from src.vector_store import VectorStore
from src.llm import OllamaLLM, MockLLM
from src.pipeline import RAGPipeline


def print_separator(char="=", length=70):
    print(char * length)


def run_demo(pipeline: RAGPipeline):
    demo_questions = [
        "What is the emergency valve code for the cooling array?",
        "What is the qubit count and decoherence time T1 of Aurora-7?",
        "What are the strategic perspectives and effects of ERP systems in organizations?",
    ]

    print_separator()
    print("RUNNING AUTOMATED DEMO QUERIES")
    print_separator()

    for idx, question in enumerate(demo_questions, start=1):
        print(f"\n[DEMO QUERY {idx}] Question: '{question}'")
        result = pipeline.query(question, top_k=2)

        print("\n--- Step 1: Retrieved Chunks (Cosine Similarity) ---")
        for rank, doc in enumerate(result["retrieved_documents"], start=1):
            print(f"[{rank}] Score: {doc['score']:.4f} | ID: {doc['id']}")
            print(f"    Excerpt: {doc['text']}")

        print("\n--- Step 2: LLM Grounded Answer ---")
        print(result["answer"])
        print_separator("-")


def interactive_mode(pipeline: RAGPipeline):
    print_separator()
    print("INTERACTIVE RAG REPL (Type 'exit' or 'quit' to stop)")
    print_separator()

    while True:
        try:
            user_query = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break

        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Exiting.")
            break

        result = pipeline.query(user_query, top_k=2)

        print("\nTop Retrieved Context:")
        for rank, doc in enumerate(result["retrieved_documents"], start=1):
            print(f"[{rank}] (Score: {doc['score']:.4f}) {doc['text']}")

        print("\nGenerated Answer:")
        print(result["answer"])
        print_separator("-")


def main():
    parser = argparse.ArgumentParser(description="Simple, Modular, Extensible RAG System")
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to file or folder containing documents (.pdf, .txt, .md) to index (default: data)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders for documents",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-minilm",
        help="Ollama embedding model (default: all-minilm)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen2.5-coder:3b",
        help="Ollama LLM model (default: qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal Q&A prompt",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute a single query and exit",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run completely offline using mock embeddings and mock LLM",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found.", file=sys.stderr)
        sys.exit(1)

    print_separator()
    print("INITIALIZING RAG PIPELINE")
    print_separator()

    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    vector_store = VectorStore()

    if args.mock:
        print("[Mode] Running with MOCK components (no Ollama required).")
        embedding_model = MockEmbedding(dimension=64)
        llm = MockLLM()
    else:
        print(f"[Mode] Ollama Backend: Embedding='{args.embed_model}', LLM='{args.llm_model}'")
        embedding_model = OllamaEmbedding(model=args.embed_model)
        llm = OllamaLLM(model=args.llm_model)

    pipeline = RAGPipeline(
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm=llm,
    )

    try:
        print(f"[Ingestion] Scanning data path: {args.data}")
        indexed_summary = pipeline.index_path(args.data, recursive=args.recursive)

        if not indexed_summary:
            print(f"[Warning] No supported documents (.pdf, .txt, .md) found in '{args.data}'.", file=sys.stderr)
        else:
            total_chunks = sum(indexed_summary.values())
            print(f"[Ingestion] Successfully indexed {len(indexed_summary)} document(s) ({total_chunks} total chunks):")
            for fpath, chunk_count in indexed_summary.items():
                print(f"  • {fpath} -> {chunk_count} chunks")
            print(f"[Vector Store] Database now contains {vector_store.count()} indexed chunks.")

        if args.query:
            print(f"\n[Query]: {args.query}")
            result = pipeline.query(args.query, top_k=2)
            print("\nAnswer:")
            print(result["answer"])
        elif args.interactive:
            interactive_mode(pipeline)
        else:
            run_demo(pipeline)
    except ConnectionError as e:
        print("\n" + "=" * 70, file=sys.stderr)
        print("ERROR: Connection to local Ollama server failed!", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"{e}\n", file=sys.stderr)
        print("How to fix:", file=sys.stderr)
        print("1. Start Ollama in a separate terminal:", file=sys.stderr)
        print("   $ ollama serve\n", file=sys.stderr)
        print("2. Or test the RAG pipeline offline with mock models:", file=sys.stderr)
        print("   $ python3 main.py --mock\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
