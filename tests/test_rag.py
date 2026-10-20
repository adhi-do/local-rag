import unittest
import math
import tempfile
import os
from src.chunker import TextChunker
from src.vector_store import VectorStore, cosine_similarity
from src.embeddings import MockEmbedding
from src.llm import MockLLM
from src.pipeline import RAGPipeline


class TestRAGComponents(unittest.TestCase):
    def test_chunker_basic(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunker = TextChunker(chunk_size=30, chunk_overlap=10)
        chunks = chunker.split_text(text, source="test_doc")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(len(chunk.text) > 0)
            self.assertEqual(chunk.source, "test_doc")

    def test_cosine_similarity(self):
        # Identical vectors -> similarity = 1.0
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)

        # Orthogonal vectors -> similarity = 0.0
        v3 = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

        # Opposite vectors -> similarity = -1.0
        v4 = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0)

        # Zero vector handled safely
        zero = [0.0, 0.0, 0.0]
        self.assertEqual(cosine_similarity(v1, zero), 0.0)

    def test_vector_store_search(self):
        store = VectorStore()
        store.add("doc1", "Document about apples", [1.0, 0.0])
        store.add("doc2", "Document about oranges", [0.0, 1.0])
        store.add("doc3", "Document about fruit", [0.707, 0.707])

        query_vec = [1.0, 0.1]
        results = store.search(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        # doc1 should have the highest similarity to [1.0, 0.1]
        self.assertEqual(results[0][0].record_id, "doc1")
        self.assertGreater(results[0][1], results[1][1])

    def test_vector_store_save_and_load(self):
        store = VectorStore()
        store.add("rec_1", "Hello world", [0.5, 0.5], metadata={"author": "Alice"})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            store.save(tmp_path)
            new_store = VectorStore()
            new_store.load(tmp_path)

            self.assertEqual(new_store.count(), 1)
            rec = new_store.records[0]
            self.assertEqual(rec.record_id, "rec_1")
            self.assertEqual(rec.text, "Hello world")
            self.assertEqual(rec.embedding, [0.5, 0.5])
            self.assertEqual(rec.metadata["author"], "Alice")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_end_to_end_mock_pipeline(self):
        embedder = MockEmbedding(dimension=32)
        llm = MockLLM()
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        store = VectorStore()

        pipeline = RAGPipeline(
            chunker=chunker,
            embedding_model=embedder,
            vector_store=store,
            llm=llm,
        )

        sample_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Quantum computers manipulate qubits to solve complex calculations. "
            "Cryogenic cooling ensures superconductivity."
        )

        num_chunks = pipeline.index_text(sample_text, source="test_sample")
        self.assertGreater(num_chunks, 0)
        self.assertEqual(store.count(), num_chunks)

        result = pipeline.query("What does cryogenic cooling do?", top_k=1)
        self.assertIn("answer", result)
        self.assertIn("retrieved_documents", result)
        self.assertEqual(len(result["retrieved_documents"]), 1)

    def test_document_loader_pdf_and_text(self):
        from src.loader import DocumentLoader

        # Test text file loading
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write("Simple text content.")
            tmp_txt = tmp.name

        try:
            docs = DocumentLoader.load_file(tmp_txt)
            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0][0], "Simple text content.")
        finally:
            if os.path.exists(tmp_txt):
                os.remove(tmp_txt)

        # Test PDF file loading if sample exists
        if os.path.exists("data/sample_document.pdf"):
            pdf_docs = DocumentLoader.load_file("data/sample_document.pdf")
            self.assertGreater(len(pdf_docs), 0)
            self.assertIn("PROJECT-NEBULA-42", pdf_docs[0][0])
            self.assertEqual(pdf_docs[0][1]["page_number"], 1)


if __name__ == "__main__":
    unittest.main()
