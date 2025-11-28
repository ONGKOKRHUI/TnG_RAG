# RefinedRAG: Advanced Retrieval with Custom Reranking

RefinedRAG is an advanced Retrieval-Augmented Generation (RAG) system designed to overcome the limitations of traditional semantic search. Unlike standard RAG implementations that rely solely on cosine similarity of the initial query, RefinedRAG employs a multi-stage retrieval process involving **follow-up question generation** and a **custom reranking algorithm**. This ensures that the retrieved context is not just semantically similar, but deeply relevant to the user's intent and any potential nuances.

## 🚀 Key Features

*   **Adaptive Contextualization:** Automatically generates follow-up questions to explore different angles of the user's query, broadening the search scope.
*   **Custom Reranking Algorithm:** Uses a weighted formula combining semantic similarity, follow-up relevance, and term-based distance (TF-IDF/Euclidean) to score documents.
*   **Multi-Provider Support:** Seamlessly switch between **OpenAI**, **Ollama** (local LLMs), and **Google Gemini**.
*   **Multi-Format Ingestion:** Supports PDF (`.pdf`) and Word (`.docx`) documents.
*   **Persistent Vector Storage:** Uses ChromaDB to store embeddings, allowing for fast retrieval without re-indexing every run.

---

## 🛠 Tech Stack

*   **Framework:** [LangChain](https://www.langchain.com/)
*   **Vector Database:** [ChromaDB](https://www.trychroma.com/)
*   **LLMs & Embeddings:** OpenAI, Ollama, Google Gemini
*   **Math & NLP:** NumPy, SciPy, Scikit-learn (for TF-IDF and Euclidean distance calculations)

---

## 🧠 How It Works: The Reranking Advantage

Traditional RAG systems typically perform a single vector search:
1.  Embed User Query.
2.  Find Top-K documents based on Cosine Similarity.
3.  Feed to LLM.

**RefinedRAG improves this by adding intelligence to the retrieval step:**

1.  **Query Expansion:** The system analyzes your query and uses an LLM to generate **2 follow-up questions**. This helps capture context that might be implicit or missing in the original prompt.
2.  **Initial Retrieval:** It retrieves a broader set of candidate documents (Top 15) using standard vector search.
3.  **Advanced Reranking:** It re-scores these 15 documents using a custom formula:

    $Score = (\alpha \times \text{Initial Similarity}) + (\beta \times \text{Follow-up Similarity}) - (\gamma \times \text{TF-IDF Distance})$

    *   **Initial Similarity ($\alpha$):** How well the doc matches your direct question.
    *   **Follow-up Similarity ($\beta$):** How well the doc matches the generated follow-up questions (capturing hidden context).
    *   **TF-IDF Distance ($\gamma$):** Uses term-frequency analysis to penalize documents that are statistically "distant" in terms of vocabulary usage, helping to filter out false positives that might share high semantic similarity but low lexical overlap.

4.  **Selection:** The top 5 documents with the highest refined scores are selected as context for the final answer.

---

## 📦 Installation

### Prerequisites

*   Python 3.8+
*   [Git](https://git-scm.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/RefinedRAG.git
cd RefinedRAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory to store your API keys. You only need to set the keys for the providers you intend to use.

```bash
# .env file

# If using OpenAI
OPENAI_API_KEY=sk-proj-...

# If using Google Gemini
GOOGLE_API_KEY=AIzaSy...

# If using Ollama, no key is usually required, but ensure the server is running.
```

#### Setting up Ollama (Optional)
If you want to run everything locally:
1.  Download and install [Ollama](https://ollama.com/).
2.  Pull the models you plan to use (e.g., Llama 3):
    ```bash
    ollama pull llama3
    ```
3.  Start the Ollama server:
    ```bash
    ollama serve
    ```

---

## ⚙️ Configuration

The application is configured directly in `main.py`. Open the file to adjust the settings at the top:

```python
# main.py

# Choose your provider: "openai", "ollama", or "gemini"
PROVIDER = "openai"

# Set the LLM model name
LLM_MODEL = "gpt-4o-mini"
# Examples: "llama3", "gemini-1.5-flash"

# Set the Embedding model name
EMBED_MODEL = "text-embedding-3-small"
# Examples: "llama3", "models/embedding-001"
```

*Note: Ensure the models you select are available for the provider you have chosen.*

---

## 🚀 Usage

### 1. Add Your Knowledge

Place your documents into the `knowledge_base/` directory.
*   Supported formats: `.pdf`, `.docx`

### 2. Run the Application

Start the chat interface:

```bash
python main.py
```

*   **First Run:** The system will detect documents in `knowledge_base/`, split them into chunks, and build the ChromaDB vector store. This may take a moment depending on the size of your data.
*   **Subsequent Runs:** It will load the existing database for faster startup.

### 3. Rebuilding the Database

If you add new files to `knowledge_base/` or want to reset the index, run with the `--rebuild` flag:

```bash
python main.py --rebuild
```

### 4. Chatting

Once ready, simply type your question when prompted. The bot will:
1.  Display the follow-up questions it generated.
2.  Show the top 5 reranked documents with their scores.
3.  Provide the final answer based on the context.

Type `exit` or `quit` to stop the application.

---

## 📊 Example Output

```text
You: How does the reranker work?

-> Generating follow-up queries...
-> Follow-ups: ['What is the formula for the custom reranker?', 'How does TF-IDF affect the score?']
-> Reranking documents...

--- 🔍 Top 5 Retrieved Contexts ---
[1] Score: 0.8234 | File: technical_spec.pdf
    Preview: The reranking algorithm uses a combination of cosine similarity...

...

Bot: The reranker operates by calculating a weighted score...
```
