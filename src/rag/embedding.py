import os
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.rag.data_loader import load_all_documents

def _log(msg: str):
    if os.environ.get("BHARAT_LAW_VERBOSE") == "1":
        print(msg)


# Process-level caching of loaded SentenceTransformer models to avoid re-loading weights
_MODEL_CACHE = {}

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        global _MODEL_CACHE
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
            _log(f"[INFO] Loaded embedding model: {model_name}")
            
        self.model = _MODEL_CACHE[model_name]

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        import re
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        _log(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        
        # Structure-Aware Legal Chunk Enrichment
        enriched_chunks = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown Act")
            act_name = os.path.splitext(os.path.basename(source))[0].replace("_", " ").replace("-", " ")
            page_num = chunk.metadata.get("page", 0) + 1
            
            # Extract mentions of Section, Article, Chapter
            citations = re.findall(r'\b(?:Section|Sec\.|Article|Art\.|Chapter|Chap\.)\s+[A-Za-z0-9\-]+', chunk.page_content, re.IGNORECASE)
            # Normalize citations
            unique_citations = []
            seen = set()
            for cit in citations:
                norm_cit = re.sub(r'\s+', ' ', cit).strip().title()
                # Normalize common abbreviations
                norm_cit = norm_cit.replace("Sec.", "Section").replace("Art.", "Article").replace("Chap.", "Chapter")
                if norm_cit.lower() not in seen:
                    seen.add(norm_cit.lower())
                    unique_citations.append(norm_cit)
            
            citation_str = ", ".join(unique_citations)
            header = f"[Act: {act_name} | Page: {page_num}"
            if citation_str:
                header += f" | Citations: {citation_str}"
            header += "] "
            
            # Prepend header to page_content
            chunk.page_content = header + chunk.page_content
            enriched_chunks.append(chunk)
            
        return enriched_chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        _log(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=(os.environ.get("BHARAT_LAW_VERBOSE") == "1"))
        _log(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)