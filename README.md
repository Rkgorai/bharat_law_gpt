# 🇮🇳 Bharat Law GPT
**An advanced Agentic AI legal assistant for Indian Law, featuring a full LangGraph architecture, Hybrid RAG pipeline, and dynamic tool execution.**

## 📌 Overview
Bharat Law GPT is an AI system designed to answer complex legal questions, compute legal math (like EMIs and fines), calculate legal deadlines, scrape the web for recent rulings, and instantly draft professional legal documents.

Unlike generic chatbots, this project uses **Agentic AI** powered by LangGraph, giving the default **Llama 4 Scout 17B** LLM autonomous access to a suite of highly specialized tools.

This makes the system:
- **More accurate** (Answers grounded in uploaded PDFs via Hybrid FAISS/BM25 Search)
- **More mathematically precise** (Executes safe Python code to calculate fines or interest instead of hallucinating)
- **More autonomous** (Dynamically asks for missing facts, searches the web, and handles full drafting workflows)
- **Highly secure** (A dedicated PII Redactor ensures sensitive names and IDs never reach external LLM endpoints, and system guardrails block non-legal queries)

Now featuring a unified **Seamless Multi-Modal Interface** and **Decoupled Architecture**:
- **🚀 High-Performance Backend:** A fully decoupled FastAPI server handles SSE streaming, agent execution, and tool orchestration.
- **💬 Unified Text & Voice Frontend:** A sleek Streamlit UI provides voice dictation, Shift+Enter multi-line text areas, and a dedicated **Interactive Markdown Editor** to review and download generated drafts.
- **⚙️ Collapsible Settings Sidebar:** Toggle models, clear histories, or choose between **☀️ Light Mode**, **🌙 Dark Mode**, or **🌐 System Theme** inside a sliding glassmorphic sidebar.

---

## 🎨 Workflows & Architecture

### 1️⃣ The Agentic AI Loop (LangGraph)
The core "brain" of the application now operates as a ReAct Agent, allowing it to think and choose tools dynamically.

```mermaid
graph TD
    User[👤 User Query] --> UI[💻 Streamlit UI]
    UI -->|API Request| S[🚀 FastAPI Server]
    S --> PII[🛡️ PII Redactor]
    PII -->|Anonymized Prompt| Agent[🧠 LangGraph Agent]
    
    Agent -->|Needs Statutory Fact?| RAG["🗄️ Hybrid RAG Search"]
    Agent -->|Needs Recent Rulings?| Web["🌐 DuckDuckGo Web Search"]
    Agent -->|Needs Number Math?| Math["🧮 Math Calculator Tool"]
    Agent -->|Needs Date Math?| Calc["📅 Deadline Calculator"]
    Agent -->|Draft Requested?| Drafter["📝 Legal Document Drafter"]
    
    RAG --> Agent
    Web --> Agent
    Math --> Agent
    Calc --> Agent
    
    Drafter -->|Outputs| S
    S -->|De-anonymize| UI
    UI -->|Renders| Editor[✏️ Interactive Editor]
    Editor --> PDF[📄 Export to PDF]
```

### 2️⃣ The Voice Interaction Loop
A hands-free voice experience for quick consultations.

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant UI as 📱 App Interface
    participant STT as 👂 Whisper (STT)
    participant Agent as 🧠 LangGraph Agent
    participant TTS as 🗣️ Edge-TTS
    
    U->>UI: 🎙️ Speaks Question
    UI->>STT: Sends Audio
    STT->>UI: Returns Transcribed Text
    UI->>Agent: Submits Query
    Agent->>Agent: Tool Execution Loop
    Agent->>UI: Returns Legal Answer
    UI->>TTS: Sends Answer Text
    TTS->>UI: Returns MP3 Audio
    UI->>U: 🔊 Auto-Plays Answer
```

---

## ✨ Features
- **Decoupled Architecture:** Clean separation of concerns with a FastAPI backend and Streamlit frontend, ready for Docker orchestration and production deployment.
- **PII Protection Pipeline:** Sensitive Indian names, Aadhaar numbers, PAN cards, and emails are aggressively anonymized into zero-shot tokens (`[AADHAAR_0]`) before sending to LLM providers, ensuring total privacy.
- **LangGraph Agentic Architecture:** The AI operates autonomously, choosing between multiple tools to solve complex, multi-step user prompts.
- **Strict Sequential Tooling:** The Llama 4 agent is hard-prompted to handle mathematical tasks (like 10% late fees or EMIs) through a safe `calculate_expression` tool first, *before* attempting to draft documents.
- **Hybrid RAG Search:** Combines semantic Vector search (FAISS) with keyword search (BM25) to deeply mine uploaded legal PDFs for exact sections and acts.
- **Web Scraping:** If a case is too recent for the local database, the AI automatically spins up `ddgs` to scrape DuckDuckGo for live headlines.
- **Automated Document Drafter:** Tells the AI to draft a rent agreement or affidavit. It will interactively gather missing facts from you, calculate necessary math, pull laws via RAG, and inject the generated draft directly into a live Streamlit text editor.
- **PDF Generation Engine:** Includes a custom rendering pipeline that parses the Markdown text editor and generates official-looking PDFs with Times New Roman typography and proper spacing.
- **Security Guardrails:** Strictly enforced system prompts ensure the AI refuses any non-legal questions (coding, general trivia).
- **Dual Interface:** Switch seamlessly between Text and Voice modes using shared memory caching to prevent LLM reload lag.

---

## 📂 Project Structure

```text
bharat_law_gpt/
│
├── legal_docs/
│   └── pdf_files/          # Raw Indian legal documents used to build the vector DB
│
├── db/
│   ├── faiss_store/        # Persistent FAISS & BM25 indexes
│   └── tts_cache/          # Persistent cross-session TTS audio cache (0ms playback!)
│
├── src/                    # Modular Architecture
│   ├── agent/              # LangGraph Agent, System Prompts, Tool Definitions
│   ├── rag/                # Hybrid Vectorstore, Embedding & Chunking Logic
│   ├── ui/                 # App Shell UI layout, styles, and custom event handlers
│   └── voice/              # Edge-TTS reading pip and Faster-Whisper transcribers
│
├── server.py               # FastAPI Backend Entrypoint (Handles SSE & Agents)
├── app_ui.py               # Unified Web Portal (Streamlit Main App Entry)
├── build_db.py             # Script to ingest PDFs and build the hybrid database
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠 Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/Rkgorai/bharat_law_gpt.git
cd bharat_law_gpt
```

### **2. Install Python dependencies**
```bash
uv pip install -r requirements.txt
```
*(We highly recommend using `uv` or creating a virtual environment).*

### **3. Install System Dependencies (Linux/Mac)**
Required for audio playback functionality.
```bash
sudo apt update
sudo apt install mpv ffmpeg
```

### **4. Add Legal PDFs**
Place your PDF files (Constitution, IPC, Acts) into:
```text
legal_docs/pdf_files/
```

### **5. Build the Database**
This step processes your PDFs and creates the FAISS/BM25 indexes.
```bash
python build_db.py
```

### **6. Set Environment Variables**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY="your-api-key-here"
```

### **7. Run the Application**
Because the system is decoupled, you must run both the backend and frontend.

**Terminal 1 (Backend):**
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
python -m streamlit run app_ui.py
```
Visit the URL shown (usually `http://localhost:8501`).

---

## 🧪 Example Prompts to Test Tools

**Test Deadline Calculator**
> "I received a legal notice today regarding a property dispute, and the law states I must file a reply within exactly 45 days. What is the exact date of my deadline?"

**Test Web Search**
> "Can you search the web for any recent Supreme Court of India rulings from this month regarding cryptocurrency?"

**Test Local RAG Database**
> "What are the specific provisions regarding maternity leave under the Maternity Benefit Act in India?"

**Test Document Drafter**
> "Please draft an 11-month residential rent agreement. The owner is Mr. ABC and the tenant is Mr. X. The monthly rent is 35,000 INR. The property address is House no 9, XYZ Street, Jamshedpur. The start date is today, the security deposit is 2 Lakh INR, and utility bills are generated monthly. Please include a strict clause that the house cannot be modified without permission."

---

## 🚧 Limitations
⚠️ **This system is NOT a substitute for professional legal advice.** It is an educational and research tool. AI can make mistakes ("hallucinations"), even with RAG and strict guardrails. Always verify with official legal sources or a qualified lawyer.

---

## 👥 Contributing
Contributions are welcome!
- Add more legal documents.
- Improve the RAG pipeline or prompts.
- Create new specialized LangGraph tools.

---

## ⭐ Support
If you find this project useful, please **star the repo ⭐**.
Your support encourages further development!
