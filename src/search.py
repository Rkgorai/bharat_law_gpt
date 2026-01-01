import os
from dotenv import load_dotenv
from src.hybrid_vectorstore import HybridVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "db/faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-8b-instant"):
        # Use hybrid vectorstore instead of regular FAISS
        self.vectorstore = HybridVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            bm25_weight=0.5,  # Equal weighting for BM25 and vector search
            vector_weight=0.5
        )
        # Load the existing index if available
        if os.path.exists(os.path.join(persist_dir, "faiss.index")):
            self.vectorstore.load()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def format_history(self, chat_history: list) -> str:
        """
        Converts the list of dictionaries into a string conversation format.
        Limits to last 6 messages to save tokens.
        """
        formatted = ""
        # Take only the last 6 messages to avoid hitting token limits
        recent_history = chat_history[-6:] 
        
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n"
        return formatted

    def search_and_summarize(self, query: str, chat_history: list = [], top_k: int = 5) -> dict:
        """
        Retrieves docs and generates an answer using History + Context.
        """
        # 1. Search Vector DB
        results = self.vectorstore.query(query, top_k=top_k)
        
        # 2. Extract Text & Sources
        texts = []
        sources = set()
        
        for r in results:
            metadata = r["metadata"]
            if not metadata: continue
            texts.append(metadata.get("text", ""))
            
            # Formatting source
            fname = os.path.basename(metadata.get("source", "Unknown"))
            page = metadata.get("page", "?")
            sources.add(f"{fname} (Pg {page})")

        context_text = "\n\n".join(texts)
        history_text = self.format_history(chat_history)
        
        # 3. Handle No Results (Optional: Let LLM handle it if history has context)
        if not context_text and not history_text:
             return {"answer": "I couldn't find legal info on that.", "sources": []}

        # 4. Enhanced Prompt with History
        prompt = f"""
        You are a specialized Legal AI Assistant for Indian Law.
        
        Task: Answer the user's latest question based on the Context and Conversation History.
        
        Guidelines:
        1. Prioritize the 'Context' for specific legal facts (Sections, Acts).
        2. Use 'Conversation History' to understand follow-up questions (e.g., if user says "What about murder?", look at previous chat).
        3. If the answer is not in the context, say you don't know. Do not make up laws.
        4. Cite the specific Acts/Sections if available.

        ---
        Conversation History:
        {history_text}
        ---
        
        New Retrieved Context (Legal Docs):
        {context_text}
        ---
        
        Latest User Question: '{query}'
        
        Answer:
        """
        
        response = self.llm.invoke([prompt])
        
        return {
            "answer": response.content,
            "sources": list(sources)
        }

if __name__ == "__main__":
    rag = RAGSearch()
    result = rag.search_and_summarize("What is the punishment for theft?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])