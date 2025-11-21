from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    print("Loading Documents ....")
    docs = load_all_documents("datas")
    print("Documents Loaded.")
    print(f"Total Documents Loaded: {len(docs)}")
    print("Storing in Vector Store ....")
    store = FaissVectorStore("faiss_store")
    print("Vector Store Done.")
    # store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    # llm_model = "moonshotai/kimi-k2-instruct-0905"
    llm_model = "llama-3.1-8b-instant"
    rag_search = RAGSearch(llm_model=llm_model)

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        summary = rag_search.search_and_summarize(query, top_k=3)
        print("Summary:", summary)