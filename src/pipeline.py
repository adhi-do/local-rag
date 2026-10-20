import os
from typing import List, Dict, Any, Optional
from src.chunker import TextChunker, Chunk
from src.embeddings import BaseEmbedding, OllamaEmbedding
from src.vector_store import VectorStore, VectorRecord
from src.llm import BaseLLM, OllamaLLM
from src.loader import DocumentLoader


DEFAULT_SYSTEM_PROMPT = """You are a helpful and precise assistant.
Answer the user's question using ONLY the provided context snippets.
If the answer cannot be found in the context, clearly state: "I cannot answer this based on the provided context."
Do not make assumptions or extrapolate beyond what is explicitly stated in the context."""


class RAGPipeline:
    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[BaseLLM] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.chunker = chunker or TextChunker()
        self.embedding_model = embedding_model or OllamaEmbedding()
        self.vector_store = vector_store or VectorStore()
        self.llm = llm or OllamaLLM()
        self.system_prompt = system_prompt

    def index_text(self, text: str, source: str = "direct_input") -> int:
        chunks: List[Chunk] = self.chunker.split_text(text, source=source)
        for chunk in chunks:
            vector = self.embedding_model.embed_text(chunk.text)
            record_id = f"{source}_{chunk.chunk_id}"
            self.vector_store.add(
                record_id=record_id,
                text=chunk.text,
                embedding=vector,
                metadata={
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata,
                },
            )
        return len(chunks)

    def index_file(self, filepath: str) -> int:
        documents = DocumentLoader.load_file(filepath)
        total_chunks = 0
        for text, doc_metadata in documents:
            chunks = self.chunker.split_text(text, source=filepath)
            for chunk in chunks:
                vector = self.embedding_model.embed_text(chunk.text)
                page_suffix = f"_p{doc_metadata['page_number']}" if "page_number" in doc_metadata else ""
                record_id = f"{filepath}{page_suffix}_{chunk.chunk_id}"
                combined_meta = {**doc_metadata, **chunk.metadata, "chunk_id": chunk.chunk_id}
                self.vector_store.add(
                    record_id=record_id,
                    text=chunk.text,
                    embedding=vector,
                    metadata=combined_meta,
                )
            total_chunks += len(chunks)
        return total_chunks

    def index_directory(self, dir_path: str, recursive: bool = False) -> Dict[str, int]:
        files = DocumentLoader.find_supported_files(dir_path, recursive=recursive)
        results: Dict[str, int] = {}
        for filepath in files:
            chunks_count = self.index_file(filepath)
            results[filepath] = chunks_count
        return results

    def index_path(self, path: str, recursive: bool = False) -> Dict[str, int]:
        if os.path.isdir(path):
            return self.index_directory(path, recursive=recursive)
        elif os.path.isfile(path):
            count = self.index_file(path)
            return {path: count}
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def retrieve(self, query: str, top_k: int = 3):
        query_vector = self.embedding_model.embed_text(query)
        return self.vector_store.search(query_vector, top_k=top_k)

    def build_prompt(self, query: str, results: List[Any]) -> str:
        context_blocks = []
        for rank, (record, score) in enumerate(results, start=1):
            source_tag = record.metadata.get("filename") or os.path.basename(record.metadata.get("source", "unknown"))
            page_info = f" (Page {record.metadata['page_number']})" if "page_number" in record.metadata else ""
            context_blocks.append(
                f"[Document {rank} | Source: {source_tag}{page_info} | Relevance: {score:.4f}]\n{record.text}"
            )

        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"Context Information:\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return prompt

    def query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        results = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, results)
        answer = self.llm.generate(prompt=prompt, system_prompt=self.system_prompt)

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": record.record_id,
                    "text": record.text,
                    "score": score,
                    "metadata": record.metadata,
                }
                for record, score in results
            ],
            "augmented_prompt": prompt,
        }
