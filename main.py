import os
import argparse
import ast
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_factory import get_llm, get_embeddings
from reranker import CustomReranker

load_dotenv()

DATA_PATH = "./knowledge_base"
DB_PATH = "./chroma_db"

# PROVIDER CONFIGURATION 
PROVIDER = "openai" # or "ollama", "gemini"
LLM_MODEL = "gpt-4o-mini" # or "gemma3:4b", "gemini-1.5-flash"
EMBED_MODEL = "text-embedding-3-small" # or "gemma3:4b", "models/embedding-001"

class AdvancedRAG:
    def __init__(self):
        self.llm = get_llm(PROVIDER, LLM_MODEL)
        self.embeddings = get_embeddings(PROVIDER, EMBED_MODEL)
        self.vector_store = None
        self.reranker = CustomReranker(self.embeddings)

    def ingest_data(self, force_rebuild=False):
        """Loads and indexes data."""
        if os.path.exists(DB_PATH) and not force_rebuild:
            print("--- Loading existing ChromaDB ---")
            self.vector_store = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        else:
            print(f"--- Creating new ChromaDB (Provider: {PROVIDER}) ---")
            documents = []
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH)
            
            for file in os.listdir(DATA_PATH):
                path = os.path.join(DATA_PATH, file)
                if file.endswith(".pdf"):
                    documents.extend(PyPDFLoader(path).load())
                elif file.endswith(".docx"):
                    documents.extend(Docx2txtLoader(path).load())
            
            if not documents:
                print("No docs found in 'data/'.")
                return

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                persist_directory=DB_PATH
            )
            print("--- Ingestion Complete ---")

    def generate_follow_up_questions(self, query):
        """
        generate 2 follow-up questions. Follow prompt engineering strategy of:
        - clearly defined roles of the LLMs
        - clear instructions
        - strict output and limitation
        - can include few-shots or CoT reasoning
        """
        prompt_text = """
        You are an AI designed to assist in improving Retrieval-Augmented Generation (RAG) by generating follow-up questions.
        Your task is to anticipate two relevant follow-up questions that provide additional context or clarity.

        To do this, follow these reasoning steps:
        1. Break down the user's query into key components.
        2. Think of additional details useful for retrieval.
        3. Formulate two insightful follow-up questions.

        Output strictly a Python list of strings, e.g., ["question 1", "question 2"]. Do not explain.

        User Query: "{user_query}"
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"user_query": query})
            # clean up markdown formatting from LLM
            cleaned_response = response.replace("```python", "").replace("```", "").strip()
            return ast.literal_eval(cleaned_response)
        except Exception as e:
            print(f"Error generating follow-ups: {e}")
            return [query] # fallback to original query

    def chat(self):
        print(f"\n--- Advanced RAG Bot ({PROVIDER}) Ready ---")
        print("Type 'exit' to quit.\n")
        
        while True:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                break

            # generate follow-up questions
            print("-> Generating follow-up queries...", end="\r")
            follow_ups = self.generate_follow_up_questions(query)
            print(f"-> Follow-ups: {follow_ups}          ")

            # initial retrieval (Get top 15 candidates)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
            initial_docs = retriever.invoke(query)

            # custom Reranking
            print("-> Reranking documents...", end="\r")
            reranked_docs, scored_results = self.reranker.rerank(
                query, follow_ups, initial_docs, 
                alpha=0.5, beta=0.1, gamma=0.01
            )
            
            # select top 5 after reranking
            top_results = scored_results[:5]
            top_docs = [item['doc'] for item in top_results]

            if not top_docs:
                print("\n[!] No relevant documents found.")
            else:
                print("\n--- 🔍 Top 5 Retrieved Contexts ---")
                for i, item in enumerate(top_results):
                    doc = item['doc']
                    score = item['score']
                    source = doc.metadata.get('source', 'Unknown')
                    # preview first 200 characters, remove newlines for cleaner output
                    content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
                    
                    print(f"[{i+1}] Score: {score:.4f} | File: {os.path.basename(source)}")
                    print(f"    Preview: {content_preview}\n")
                print("-----------------------------------\n")

            # generate answer
            context_text = "\n\n".join([d.page_content for d in top_docs])
            
            rag_prompt = f"""
            Answer the question based ONLY on the context below.
            
            Context:
            {context_text}
            
            Question: {query}
            """
            
            print("-> Generating answer...      ", end="\r")
            response = self.llm.invoke(rag_prompt)
            
            print(f"\nBot: {response.content}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector DB")
    args = parser.parse_args()

    app = AdvancedRAG()
    app.ingest_data(force_rebuild=args.rebuild)
    app.chat()