import shutil
import os
from src.rag.data_loader import load_all_documents
from src.rag.hybrid_vectorstore import HybridVectorStore

DATA_DIR = "legal_docs"
DB_PATH = "db/faiss_store"

def clean_rebuild():
    print("⚠️  Starting Full Database Rebuild with Hybrid Search (BM25 + Vector)...")
    
    # 1. Clear old database
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"🗑️  Deleted old database at {DB_PATH}")
    
    # 2. Load Documents
    print(f"📂 Loading PDF documents from {DATA_DIR}...")
    docs = load_all_documents(DATA_DIR)
    
    if not docs:
        print("❌ Error: No documents found. Please add PDFs to 'legal_docs/'")
        return

    # 3. Build New Hybrid Index (BM25 + Vector Embeddings)
    print(f"🏗️  Creating Hybrid Search Index (BM25 + Embeddings) for {len(docs)} document chunks...")
    store = HybridVectorStore(
        persist_dir=DB_PATH,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    store.build_from_documents(docs)
    
    print("✅ Hybrid Database Rebuild Complete!")
    print("📊 Search now uses BM25 (keyword) + Vector (semantic) for better results!")

if __name__ == "__main__":
    clean_rebuild()