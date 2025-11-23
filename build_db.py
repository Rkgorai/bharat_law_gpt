import shutil
import os
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

DATA_DIR = "legal_docs"
DB_PATH = "db/faiss_store"

def clean_rebuild():
    print("⚠️  Starting Full Database Rebuild...")
    
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

    # 3. Build New Index (This will now include Metadata!)
    print(f"🏗️  Creating Embeddings for {len(docs)} document chunks...")
    store = FaissVectorStore(persist_dir=DB_PATH)
    store.build_from_documents(docs)
    
    print("✅ Database Rebuild Complete!")

if __name__ == "__main__":
    clean_rebuild()