import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "db/faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load the existing index if available
        if os.path.exists(os.path.join(persist_dir, "faiss.index")):
            self.vectorstore.load()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> dict:
        """
        Retrieves docs and generates an answer with sources.
        Returns a dictionary: {"answer": str, "sources": list}
        """
        # 1. Search Vector DB
        results = self.vectorstore.query(query, top_k=top_k)
        
        # 2. Extract Text & Sources
        texts = []
        sources = set() # Use set to avoid duplicates
        
        for r in results:
            metadata = r["metadata"]
            if not metadata:
                continue
                
            # Append Text
            texts.append(metadata.get("text", ""))
            
            # Extract Source (Filename + Page Number)
            file_path = metadata.get("source", "Unknown Document")
            file_name = os.path.basename(file_path) # Get just 'IPC.pdf' not '/data/IPC.pdf'
            page = metadata.get("page", "N/A")
            
            sources.add(f"{file_name} (Page {page})")

        context = "\n\n".join(texts)
        
        # 3. Handle No Results
        if not context:
            return {
                "answer": "I could not find any relevant legal documents in the database regarding this query.",
                "sources": []
            }

        # 4. Generate Answer with improved Legal Prompt
        prompt = f"""
        You are a specialized Legal AI Assistant for Indian Law.
        Answer the user's query strictly based on the context provided below.
        
        Guidelines:
        1. Be precise and factual.
        2. Reference specific Acts or Sections if mentioned in the context.
        3. Do not hallucinate or make up laws not present in the text.
        
        Query: '{query}'
        
        Context:
        {context}
        
        Answer:
        """
        
        response = self.llm.invoke([prompt])
        
        # 5. Return Structured Output
        return {
            "answer": response.content,
            "sources": list(sources)
        }

if __name__ == "__main__":
    rag = RAGSearch()
    result = rag.search_and_summarize("What is the punishment for theft?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])