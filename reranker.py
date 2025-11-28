import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean
from scipy.special import expit  # Sigmoid function
from langchain_core.documents import Document

class CustomReranker:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def compute_normalized_distances(self, question, relevant_chunks):
        """
        Replicates your notebook's logic: TF-IDF -> Euclidean -> Sigmoid.
        """
        if not relevant_chunks:
            return []

        # Combine question and chunks into a single list for vectorization
        chunk_texts = [chunk.page_content for chunk in relevant_chunks]
        all_texts = [question] + chunk_texts

        # Convert texts to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts).toarray()
        except ValueError:
            # Handle cases where vocabulary is empty or texts are empty
            return [{"chunk": c, "distance": 1.0} for c in chunk_texts]

        # Compute distances
        question_vector = tfidf_matrix[0]
        chunk_vectors = tfidf_matrix[1:]

        distances = [euclidean(question_vector, cv) for cv in chunk_vectors]

        # Normalize distances using sigmoid
        normalized_distances = expit(distances)

        return normalized_distances

    def compute_similarity(self, vector1, vector2):
        """Compute cosine similarity."""
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vector1, vector2) / (norm1 * norm2)

    def rerank(self, question, follow_up_questions, retrieved_docs, alpha=0.5, beta=0.1, gamma=0.01):
        """
        Re-scores documents based on the notebook formula:
        Score = alpha * init_sim + beta * follow_up_sim - gamma * dist
        """
        if not retrieved_docs:
            return []

        # 1. Calculate TF-IDF Distances
        distances = self.compute_normalized_distances(question, retrieved_docs)

        # 2. Get Embeddings (We re-embed here to ensure we have the vectors for math)
        # Note: In production with millions of docs, you'd fetch vectors from DB. 
        # For <20 docs, re-embedding on the fly is fast.
        initial_query_embedding = self.embedding_model.embed_query(question)
        follow_up_embeddings = [self.embedding_model.embed_query(fq) for fq in follow_up_questions]
        doc_embeddings = self.embedding_model.embed_documents([d.page_content for d in retrieved_docs])

        scored_results = []

        for idx, doc in enumerate(retrieved_docs):
            # A. Initial Similarity
            init_score = self.compute_similarity(initial_query_embedding, doc_embeddings[idx])

            # B. Follow-up Similarity (Sum of sims)
            fu_score = sum(
                self.compute_similarity(fu_emb, doc_embeddings[idx]) 
                for fu_emb in follow_up_embeddings
            )

            # C. Distance from TF-IDF
            dist = distances[idx]

            # D. Final Formula from Notebook
            final_score = (alpha * init_score) + (beta * fu_score) - (gamma * dist)

            scored_results.append({
                "doc": doc,
                "score": final_score,
                "debug_info": {
                    "init_score": round(init_score, 3),
                    "fu_score": round(fu_score, 3),
                    "dist": round(dist, 3)
                }
            })

        # Sort by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return just the docs, sorted
        return [item["doc"] for item in scored_results], scored_results