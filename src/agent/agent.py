import os
import datetime
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from ddgs import DDGS
from src.rag.hybrid_vectorstore import HybridVectorStore

# Dependency for local RAG
class LocalSearchWrapper:
    def __init__(self, persist_dir="db/faiss_store", embedding_model="all-MiniLM-L6-v2"):
        self.vectorstore = HybridVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            bm25_weight=0.5,
            vector_weight=0.5
        )
        if os.path.exists(os.path.join(persist_dir, "faiss.index")):
            self.vectorstore.load()

    def search(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = []
        for r in results:
            metadata = r.get("metadata", {})
            if metadata:
                texts.append(metadata.get("text", ""))
        if not texts:
            return "No relevant information found in the local legal database."
        return "\n\n".join(texts)

_local_search = None

@tool
def local_rag_search(query: str) -> str:
    """
    Search the local Bharat Law database for Indian legal documents, bare acts, IPC sections, and legal definitions.
    Use this tool FIRST for any question about Indian laws or legal facts.
    """
    global _local_search
    if _local_search is None:
        _local_search = LocalSearchWrapper()
    return _local_search.search(query)

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent case laws, news, or legal updates that might not be in the local database.
    Use this if the local_rag_search does not yield sufficient information.
    """
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "No results found on the web."
        
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
            
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching the web: {str(e)}"

@tool
def calculate_deadline(start_date: str, days: str) -> str:
    """
    Calculate a legal deadline by adding or subtracting days from a start date.
    start_date: The date in 'YYYY-MM-DD' format.
    days: Number of days to add (positive) or subtract (negative). Pass this as a string, e.g. "45".
    """
    try:
        dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        new_dt = dt + datetime.timedelta(days=int(days))
        return new_dt.strftime("%Y-%m-%d")
    except Exception as e:
        return f"Error calculating date: {str(e)}"

@tool
def calculate_expression(expression: str) -> str:
    """
    Evaluate a basic mathematical expression safely.
    Use this tool whenever you need to compute numbers, such as loan installments, EMIs, compound interest, or total repayment amounts.
    expression: A mathematical expression using numbers and operators (e.g. "500000 * 0.10 / 12" or "500000 * (0.01) * (1.01)**12 / ((1.01)**12 - 1)").
    """
    import re
    # Strictly allow digits, decimal points, spaces, parentheses, and arithmetic operators: +, -, *, /, **
    clean_expr = expression.replace(" ", "")
    if not re.match(r'^[0-9+\-*/().\s*]+$', clean_expr):
        return "Error: Invalid characters in mathematical expression. Only numbers and basic arithmetic operators (+, -, *, /, **) are allowed."
    try:
        # Evaluate safely without builtins
        result = eval(clean_expr, {"__builtins__": None}, {})
        if isinstance(result, float):
            return f"{result:.2f}"
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def draft_document(document_type: str, details: str) -> str:
    """
    Generate a legal document. 
    Use this tool ONLY when you have collected all necessary details from the user.
    document_type: The type of document to draft (e.g., "Rent Agreement").
    details: All the gathered facts, names, addresses, and clauses.
    """
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="meta-llama/llama-4-scout-17b-16e-instruct")
    prompt = (
        f"You are an expert legal drafter specializing in Indian Banking and Financial Laws. "
        f"Your task is to draft a highly professional, comprehensive, and legally binding {document_type} "
        f"by fully integrating the following user-provided details:\n"
        f"=== USER-PROVIDED DETAILS ===\n{details}\n=============================\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. You MUST substitute every user-provided detail directly into the appropriate sections of the draft. "
        "Do NOT output literal curly-brace template placeholders, bracketed placeholders, or developer variables "
        "(e.g., do NOT write `{borrower_name}`, `{borrower_address}`, `{loan_amount}`, `[DATE]`, `{interest_rate}`, etc.).\n"
        "2. If a specific detail required for the draft is missing from the provided input, use a clean blank line "
        "like '_________________' or make a realistic default assumption if appropriate. NEVER output curly-brace placeholder variables.\n"
        "3. Ensure the document is written in a professional, legal tone, customized to the details provided.\n"
        "4. FINANCIAL TABLES: If the document involves a repayment cycle, EMI, or amortization, "
        "you MUST present the full payment cycle as a beautifully formatted markdown table.\n"
        "5. DOMAIN-SPECIFIC LAWS: You MUST incorporate the relevant Indian laws for the specific document type (e.g., for loans, cite RBI guidelines, SARFAESI Act, CIBIL; for rent, cite the applicable State Rent Control Act; for employment, cite labor laws).\n"
        "6. Output ONLY the complete, ready-to-use drafted document text. Do not include any introductory preambles or postscripts."
    )
    response = llm.invoke(prompt)
    
    # Save the huge draft text to a file so it doesn't crash the Groq tool parser
    with open("db/latest_draft.txt", "w") as f:
        f.write(response.content)
        
    return "<DRAFT_SAVED> Successfully drafted! Tell the user the draft is ready in the editor."

tools = [local_rag_search, web_search, calculate_deadline, draft_document, calculate_expression]

def create_legal_agent(llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
    
    current_date_str = datetime.date.today().strftime("%Y-%m-%d")
    
    prompt_text = f"""You are a specialized Legal AI Assistant for Indian Law.
Today's date is: {current_date_str}.

CRITICAL BEHAVIOR RULES:
1. MATHEMATICAL CALCULATIONS (EMI, FINES, RENT INCREASES, ETC.):
   - If the task requires ANY math (e.g. calculating EMIs, 10% late fee fines, compound interest), you MUST NEVER calculate it in your head.
   - You MUST FIRST call the `calculate_expression` tool with the exact Python math formula as the SOLE tool in your first turn.
     * Example for EMI: P * r * (1 + r)**n / ((1 + r)**n - 1) (where r is monthly interest rate, n is total months).
     * Example for 10% fine: Amount * 0.10
   - ONLY after receiving the calculated value from the tool can you proceed to draft the document. Do NOT call `draft_document` in the same turn as `calculate_expression`.

2. DOCUMENT DRAFTING:
   - When asked to draft ANY legal agreement, contract, or document (Rent, Loan, NDA, Notice, etc.), you MUST use the `draft_document` tool.
   - Never write the document text in the chat window yourself.
   - Pass all collected details and tool-calculated math into the `draft_document` tool's `details` argument.

3. SEARCH RULE:
   - Search the legal database using `local_rag_search` or `web_search` to find relevant Indian laws, rules, or regulations before drafting.

4. SEQUENTIAL EXECUTION:
   - Always run your steps sequentially:
     * Turn 1: Call `calculate_expression` (if math is needed) and `local_rag_search` (to find relevant laws).
     * Turn 2: Once you receive the tool responses, call `draft_document` with all details.
     * Turn 3: Inform the user that the document has been successfully drafted and is ready in the editor.

5. GENERAL GUARDRAIL:
   - Refuse non-legal questions by replying: "I am a Legal AI Assistant specialized in Indian Law. I cannot assist with non-legal queries."
"""

    agent = create_react_agent(llm, tools, prompt=prompt_text)
    return agent

