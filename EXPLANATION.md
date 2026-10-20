# Complete Step-by-Step Guide: How This RAG System Works (100% Breakdown)

This document provides a complete, clear, and comprehensive breakdown of the Retrieval-Augmented Generation (RAG) system built in this project. Every mathematical concept, architecture decision, data transformation, and code component is explained from first principles.

---

## 1. What is RAG and Why Do We Need It?

### The Core Problem of Large Language Models (LLMs)
Large Language Models like GPT-4 or Qwen possess extensive general knowledge, but they have three major limitations:
1. **Knowledge Cutoff**: They cannot know events or publications that occurred after their training date.
2. **Private / Domain Knowledge**: They have zero access to your internal documentation, proprietary databases, private files, or company protocols.
3. **Hallucination**: When an LLM does not know a specific fact, it often fabricates a plausible-sounding but completely incorrect response.

### Why Not Just Fine-Tune?
* **Fine-Tuning is Expensive & Slow**: Training requires high-end GPUs, significant time, and curated datasets.
* **Knowledge Decay**: When documents change (e.g., a policy update), you must re-train the model.
* **Unverifiable**: You cannot easily see *why* a model said what it said or cite the exact source document.

### The RAG Solution
**RAG (Retrieval-Augmented Generation)** solves this by treating the LLM like an open-book student:
1. When asked a question, we **retrieve** only the specific paragraphs from our documents that are relevant to that question.
2. We **augment** the prompt by pasting those retrieved paragraphs into the context window.
3. We let the LLM **generate** an answer strictly grounded in those retrieved facts.

---

## 2. System Architecture & Lifecycle

The pipeline operates in two distinct phases:

### Phase A: Ingestion (Indexing Phase)
This happens whenever new documents are added to the knowledge base:
```
[Raw Document (Text/PDF)]
         │
         ▼
[1. Text Chunker]  ──> Splits text into overlapping semantic blocks
         │
         ▼
[2. Embedding Model] ──> Converts each text chunk into a high-dimensional vector
         │
         ▼
[3. Vector Store] ──> Stores (Chunk Text + Vector + Metadata) in an index
```

### Phase B: Query & Generation (Inference Phase)
This happens in real time whenever a user asks a question:
```
[User Question: "What is the emergency valve code?"]
         │
         ▼
[Embedding Model] ──> Converts question into a query vector
         │
         ▼
[Vector Store Search] ──> Computes Cosine Similarity against all stored chunks
         │
         ▼
[Top-K Retrieval] ──> Extracts the most similar chunks (e.g., top 2)
         │
         ▼
[Prompt Augmentation] ──> Combines System Prompt + Retrieved Chunks + Question
         │
         ▼
[Generative LLM] ──> Generates a factually grounded answer
         │
         ▼
[Final Output to User]
```

---

## 3. Deep Dive into Every Component

### Step 0: Document Ingestion & PDF Parsing (`src/loader.py`)

#### Handling Plain Text vs. PDFs
Raw text files (`.txt`, `.md`) can be read directly as UTF-8 character streams. However, **PDF (Portable Document Format)** files are compressed binary streams storing font glyphs, layout commands, and graphics objects.

`DocumentLoader` handles this distinction automatically:
* **Plain Text**: Reads standard UTF-8 text files.
* **PDFs**: Uses `pypdf.PdfReader` to extract text page-by-page. It attaches metadata like `page_number` and `total_pages` to each page.
* **Page-Aware Citations**: Because page numbers are preserved across chunks, the prompt augmentation layer can cite the exact page number (e.g. `[Document 1 | Source: manual.pdf (Page 4)]`) when answering.

### Step 1: Text Chunking (`src/chunker.py`)

#### Why Chunking is Necessary
* **Context Preservation**: Embedding an entire 100-page book as a single vector averages out all information into a blurry, unusable representation. Small, focused chunks capture distinct concepts sharply.
* **Token Limits**: Models have input size limits. By chunking text into pieces of ~400–500 characters, retrieval finds the exact needle in the haystack.

#### Why Chunk Overlap Matters
Consider the text:
> "In case of fire, use the east staircase. The west staircase is strictly reserved for medical emergency teams."

If a chunker blindly splits between "east staircase." and "The west staircase", a chunk might end with:
* Chunk 1: *"In case of fire, use the east staircase."*
* Chunk 2: *"is strictly reserved for medical emergency teams."*

Chunk 2 loses the subject ("The west staircase"). With **chunk overlap** (e.g., 100 characters), the end of Chunk 1 is repeated at the beginning of Chunk 2, ensuring no idea or sentence is severed at a boundary.

#### How Our Chunker Works
1. `_split_into_sentences`: Splits the raw text using regular expressions on paragraph breaks (`\n\n`) and sentence endings (`. `, `! `, `? `).
2. It accumulates sentences into a chunk until adding another sentence would exceed `chunk_size` (default: 500 characters).
3. When saving a chunk, it calculates the trailing sentences that fit within `chunk_overlap` (default: 100 characters) and seeds the next chunk with them.
4. Each chunk is saved as a `Chunk` object containing:
   * `text`: The text segment.
   * `chunk_id`: Integer identifier.
   * `source`: Originating file path.
   * `metadata`: Character counts and offsets.

---

### Step 2: Vector Embeddings (`src/embeddings.py`)

#### What is a Vector Embedding?
An embedding model is a neural network trained to map human language into a continuous numerical vector space (an array of floating point numbers, e.g., 384 dimensions in `all-minilm`).

In this vector space:
* Words and sentences with **similar meanings** are positioned **geometrically close together**.
* Sentences with unrelated meanings are positioned far apart or orthogonal to each other.

For example:
* `"The reactor temperature is rising"` and `"The core is getting hot"` will produce vectors pointing in nearly the exact same direction.
* `"A recipe for baking chocolate cake"` will point in an entirely different direction.

#### Interface Design & Extensibility
In `src/embeddings.py`, we define `BaseEmbedding` using Python's `abc.ABC`:
```python
class BaseEmbedding(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
```
This design allows hot-swapping any embedding provider without modifying the rest of the application:
1. **`OllamaEmbedding`**: Sends an HTTP POST request to Ollama's `/api/embeddings` endpoint with the text and model name (`all-minilm`).
2. **`MockEmbedding`**: A deterministic fallback that produces unit-normalized vectors using word hashes, requiring zero external services or dependencies.

---

### Step 3: Vector Store & Cosine Similarity (`src/vector_store.py`)

#### The Mathematics of Cosine Similarity
To find the chunks most relevant to a user query, we measure the angle $\theta$ between the query vector $\mathbf{A}$ and each document vector $\mathbf{B}$.

The mathematical formula is:
$$\text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

#### Step-by-Step Numerical Example
Suppose we have 2-dimensional vectors:
* Query vector $\mathbf{A} = [1.0, 2.0]$
* Document vector $\mathbf{B} = [2.0, 3.0]$

1. **Calculate Dot Product ($\mathbf{A} \cdot \mathbf{B}$)**:
   $$\mathbf{A} \cdot \mathbf{B} = (1.0 \times 2.0) + (2.0 \times 3.0) = 2.0 + 6.0 = 8.0$$

2. **Calculate Euclidean Norms (Lengths)**:
   $$\|\mathbf{A}\| = \sqrt{1.0^2 + 2.0^2} = \sqrt{1 + 4} = \sqrt{5} \approx 2.236$$
   $$\|\mathbf{B}\| = \sqrt{2.0^2 + 3.0^2} = \sqrt{4 + 9} = \sqrt{13} \approx 3.606$$

3. **Compute Cosine Similarity**:
   $$\text{Similarity} = \frac{8.0}{2.236 \times 3.606} = \frac{8.0}{8.062} \approx 0.9923$$

#### Interpreting the Score:
* **$1.0$**: Vectors point in identical directions (identical meaning).
* **$0.0$**: Vectors are orthogonal (completely unrelated).
* **$-1.0$**: Vectors point in opposite directions.

#### Why Cosine Similarity Instead of Euclidean Distance?
Euclidean distance measures the straight-line distance between vector tips. If one chunk is short (few words) and another is long (many words), their vector magnitudes can differ greatly even if their semantic topic is identical. Cosine similarity normalizes for length by only measuring the **angle** (direction) of meaning.

#### Storage and Persistence
The `VectorStore` maintains an in-memory list of `VectorRecord` items. It also provides `save(filepath)` and `load(filepath)` methods to serialize the entire vector database into human-readable JSON.

---

### Step 4: The Generative LLM Client (`src/llm.py`)

The LLM is responsible for synthesizing the retrieved facts into fluent, natural language.

#### Key Parameters:
* **Model**: `qwen2.5-coder:3b` (fast, lightweight, highly accurate instruction follower).
* **Temperature**: Set to `0.2`. A low temperature makes the model deterministic and strictly factual, preventing creative drift or hallucination.

Like the embedding model, `src/llm.py` defines `BaseLLM`, allowing easy substitution of any backend (Ollama, OpenAI, Anthropic, Gemini, or local models).

---

### Step 5: The RAG Pipeline Orchestrator (`src/pipeline.py`)

`RAGPipeline` ties all individual components into an end-to-end workflow:

1. **`index_file(filepath)`**:
   * Reads the text file.
   * Calls `chunker.split_text()`.
   * For each chunk, computes its embedding vector.
   * Stores the chunk text, vector, and metadata in `vector_store`.

2. **`retrieve(query, top_k)`**:
   * Computes the vector embedding of the user's question.
   * Searches the `vector_store` using cosine similarity.
   * Returns the top $K$ records with their numerical similarity scores.

3. **`build_prompt(query, results)`**:
   Formats the retrieved passages into a structured prompt:
   ```text
   Context Information:
   ---------------------
   [Document 1 | Source: data/knowledge_base.txt | Relevance: 0.5257]
   If temperature exceeds 22.0 millikelvins... the manual emergency override
   valve is located on Sub-level 2 and bears the identification code VALVE-OMEGA-9.

   [Document 2 | Source: data/knowledge_base.txt | Relevance: 0.4065]
   The Helios Cooling Array maintains the operational thermal envelope...
   ---------------------

   Question: What is the emergency valve code for the cooling array?

   Answer:
   ```

4. **`query(query_text, top_k)`**:
   * Passes the augmented prompt and strict system instructions to `llm.generate()`.
   * System Prompt:
     > *"You are a helpful and precise assistant. Answer the user's question using ONLY the provided context snippets. If the answer cannot be found in the context, clearly state: 'I cannot answer this based on the provided context.' Do not make assumptions or extrapolate beyond what is explicitly stated in the context."*
   * Returns a complete dictionary containing the answer, retrieved documents, similarity scores, and full prompt.

---

## 4. End-to-End Walkthrough of a Real Query

Let's trace what happens when we ask:
> **Question**: *"What is the emergency valve code for the cooling array?"*

1. **Embedding**:
   * Query string: `"What is the emergency valve code for the cooling array?"`
   * Passed to `all-minilm` $\rightarrow$ Outputs a 384-dimensional vector: `[0.021, -0.054, 0.112, ...]`.

2. **Similarity Scoring**:
   * Vector store compares this 384-dim vector against all 8 stored chunk vectors using `cosine_similarity`.
   * Chunk 1 (Section 1 & 2 boundary) gets the highest score: **`0.5257`**.
   * Chunk 0 (Helios Cooling Array overview) gets the second highest score: **`0.4065`**.
   * Other chunks (e.g., Section 5 fiber optics) score significantly lower (~`0.18`).

3. **Context Injection**:
   * Chunk 1 contains: `"...the manual emergency override valve is located on Sub-level 2 and bears the identification code VALVE-OMEGA-9."`
   * This chunk is injected into the prompt.

4. **Inference & Grounding**:
   * LLM reads the context and prompt.
   * LLM outputs:
     > **"The emergency valve code for the cooling array is VALVE-OMEGA-9."**

5. **Hallucination Prevention**:
   * If we ask: *"What is the capital of Mars?"*
   * Retrieved chunks have low relevance scores.
   * The LLM, constrained by the system prompt, responds:
     > **"I cannot answer this based on the provided context."**

---

## 5. How to Extend This System

The code was intentionally designed with pure interfaces and zero framework lock-in. Here is how you can extend each part:

### A. Swapping Embeddings to OpenAI
Create a new class in `src/embeddings.py`:
```python
import openai

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding
```

### B. Swapping LLM to OpenAI or Claude
In `src/llm.py`:
```python
class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        res = self.client.chat.completions.create(model=self.model, messages=messages)
        return res.choices[0].message.content
```

### C. Scaling to Millions of Documents
For millions of vectors:
* Replace the linear loop in `VectorStore.search` with an Approximate Nearest Neighbor (ANN) index like **FAISS** or **HNSWLib**.
* The rest of your chunking, prompt building, and LLM code remains 100% identical!
