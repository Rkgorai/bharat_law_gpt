import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.embedding import EmbeddingPipeline

class HybridVectorStore:
    """
    Hybrid search combining BM25 (keyword-based) and FAISS (semantic vector search).
    Uses a weighted combination of both retrieval methods for better results.
    """
    
    def __init__(
        self, 
        persist_dir: str = "db/faiss_store", 
        embedding_model: str = "all-MiniLM-L6-v2", 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ):
        """
        Initialize hybrid search system.
        
        Args:
            persist_dir: Directory to save/load the database
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Vector search components
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        
        # BM25 components
        self.bm25 = None
        self.tokenized_corpus = []
        self.corpus_texts = []
        
        # Chunking settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Hybrid weights
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        print(f"[INFO] Initialized Hybrid Vector Store with embedding model: {embedding_model}")
        print(f"[INFO] BM25 weight: {bm25_weight}, Vector weight: {vector_weight}")

    def build_from_documents(self, documents: List[Any]):
        """Build both BM25 and vector indexes from documents."""
        print(f"[INFO] Building hybrid store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        # 1. Chunk documents
        chunks = emb_pipe.chunk_documents(documents)
        print(f"[INFO] Created {len(chunks)} chunks")
        
        # 2. Generate embeddings for vector search
        embeddings = emb_pipe.embed_chunks(chunks)
        
        # 3. Prepare metadata and texts
        metadatas = []
        self.corpus_texts = []
        
        for chunk in chunks:
            meta = chunk.metadata.copy() if chunk.metadata else {}
            meta["text"] = chunk.page_content
            metadatas.append(meta)
            self.corpus_texts.append(chunk.page_content)
        
        # 4. Build BM25 index
        print("[INFO] Building BM25 index...")
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[INFO] BM25 index built with {len(self.tokenized_corpus)} documents")
        
        # 5. Build FAISS vector index
        print("[INFO] Building FAISS vector index...")
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        
        # 6. Save everything
        self.save()
        print(f"[INFO] Hybrid store built and saved to {self.persist_dir}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """Add embeddings to FAISS index."""
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index")

    def save(self):
        """Save all components to disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        bm25_path = os.path.join(self.persist_dir, "bm25.pkl")
        corpus_path = os.path.join(self.persist_dir, "corpus.pkl")
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, faiss_path)
        
        # Save metadata
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save BM25 and corpus
        with open(bm25_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        
        with open(corpus_path, "wb") as f:
            pickle.dump(self.corpus_texts, f)
        
        print(f"[INFO] Saved hybrid store (FAISS + BM25) to {self.persist_dir}")

    def load(self):
        """Load all components from disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        bm25_path = os.path.join(self.persist_dir, "bm25.pkl")
        corpus_path = os.path.join(self.persist_dir, "corpus.pkl")
        
        # Load FAISS index
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
            print(f"[INFO] Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] Loaded {len(self.metadata)} metadata entries")
        
        # Load BM25
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
                self.bm25 = bm25_data["bm25"]
                self.tokenized_corpus = bm25_data["tokenized_corpus"]
            print(f"[INFO] Loaded BM25 index with {len(self.tokenized_corpus)} documents")
        
        # Load corpus texts
        if os.path.exists(corpus_path):
            with open(corpus_path, "rb") as f:
                self.corpus_texts = pickle.load(f)

    def _vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Perform vector-based search using FAISS."""
        if self.index is None:
            return []
        
        D, I = self.index.search(query_embedding, top_k)
        results = []
        
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                results.append({
                    "index": int(idx),
                    "score": float(1 / (1 + dist)),  # Convert distance to similarity score
                    "metadata": self.metadata[idx]
                })
        
        return results

    def _bm25_search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Perform BM25 keyword search."""
        if self.bm25 is None:
            return []
        
        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                results.append({
                    "index": int(idx),
                    "score": float(scores[idx]),
                    "metadata": self.metadata[idx]
                })
        
        return results

    def hybrid_search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            
        Returns:
            List of results with combined scores
        """
        print(f"[INFO] Performing hybrid search for: '{query_text}'")
        
        # Get more candidates from each method
        candidate_k = top_k * 3
        
        # 1. Vector search
        query_emb = self.model.encode([query_text]).astype('float32')
        vector_results = self._vector_search(query_emb, top_k=candidate_k)
        
        # 2. BM25 search
        bm25_results = self._bm25_search(query_text, top_k=candidate_k)
        
        # 3. Combine scores
        combined_scores = {}
        
        # Normalize and add vector scores
        if vector_results:
            max_vector_score = max(r["score"] for r in vector_results)
            min_vector_score = min(r["score"] for r in vector_results)
            vector_range = max_vector_score - min_vector_score if max_vector_score != min_vector_score else 1
            
            for result in vector_results:
                idx = result["index"]
                normalized_score = (result["score"] - min_vector_score) / vector_range
                combined_scores[idx] = {
                    "score": normalized_score * self.vector_weight,
                    "metadata": result["metadata"]
                }
        
        # Normalize and add BM25 scores
        if bm25_results:
            max_bm25_score = max(r["score"] for r in bm25_results)
            min_bm25_score = min(r["score"] for r in bm25_results)
            bm25_range = max_bm25_score - min_bm25_score if max_bm25_score != min_bm25_score else 1
            
            for result in bm25_results:
                idx = result["index"]
                normalized_score = (result["score"] - min_bm25_score) / bm25_range
                
                if idx in combined_scores:
                    combined_scores[idx]["score"] += normalized_score * self.bm25_weight
                else:
                    combined_scores[idx] = {
                        "score": normalized_score * self.bm25_weight,
                        "metadata": result["metadata"]
                    }
        
        # 4. Sort by combined score and return top k
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1]["score"], 
            reverse=True
        )[:top_k]
        
        final_results = []
        for idx, data in sorted_results:
            final_results.append({
                "index": idx,
                "score": data["score"],
                "metadata": data["metadata"]
            })
        
        print(f"[INFO] Hybrid search returned {len(final_results)} results")
        return final_results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Main query method that uses hybrid search."""
        return self.hybrid_search(query_text, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    
    # Build hybrid store
    docs = load_all_documents("legal_docs")
    store = HybridVectorStore(
        persist_dir="db/faiss_store",
        bm25_weight=0.5,
        vector_weight=0.5
    )
    store.build_from_documents(docs)
    
    # Load and query
    store.load()
    results = store.query("What is the punishment for theft?", top_k=3)
    
    print("\n=== Search Results ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Text: {result['metadata'].get('text', '')[:200]}...")
