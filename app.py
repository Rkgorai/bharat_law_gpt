import os
import sys
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Configuration Paths
DATA_DIR = "legal_docs"         # Folder containing your PDFs
DB_PATH = "db/faiss_store"      # Folder where vector index will be saved

def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs(DB_PATH, exist_ok=True)

def main():
    print("--- Bharat Law GPT Initializing ---")
    ensure_directories()
    
    # Initialize Vector Store Wrapper
    store = FaissVectorStore(persist_dir=DB_PATH)
    
    # specific file to check if DB exists
    index_file = os.path.join(DB_PATH, "faiss.index")

    if os.path.exists(index_file):
        print(f"[INFO] Vector store found at '{DB_PATH}'. Loading...")
        store.load()
    else:
        print(f"[WARN] No vector store found at '{DB_PATH}'. Building from scratch...")
        print(f"[INFO] Loading documents from '{DATA_DIR}'...")
        
        # Load documents
        docs = load_all_documents(DATA_DIR)
        
        if not docs:
            print("[ERROR] No documents found! Check your 'legal_docs' folder.")
            return

        print(f"[INFO] Loaded {len(docs)} documents. Creating embeddings...")
        
        # Build and Save
        store.build_from_documents(docs)
        print("[INFO] Vector store built and saved successfully.")

    # Initialize Search Engine
    # CRITICAL: We pass DB_PATH so RAGSearch knows where to look
    llm_model = "llama-3.1-8b-instant"
    print(f"[INFO] Initializing RAG Search with LLM: {llm_model}")
    
    rag_search = RAGSearch(persist_dir=DB_PATH, llm_model=llm_model)

    print("\n--- System Ready ---")
    while True:
        try:
            query = input("\nEnter your legal query (or 'exit' to quit): ").strip()
            if query.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            if not query:
                continue
                
            print("\n🔍 Searching and Summarizing...")
            summary = rag_search.search_and_summarize(query, top_k=3)
            print(f"\n📝 **Summary**:\n{summary}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()