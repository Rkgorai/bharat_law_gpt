# 🇮🇳 Bharat Law GPT
**An AI-powered legal assistant for Indian Law using a complete Retrieval-Augmented Generation (RAG) pipeline.**

## 📌 Overview
Bharat Law GPT is an AI system designed to answer questions related to Indian laws, IPC sections, legal definitions, procedures, and more.
Unlike generic chatbots, this project uses **Retrieval-Augmented Generation (RAG)** to ground all answers on real legal documents.

This makes the system:
- More accurate  
- More explainable  
- Less hallucination-prone  
- More useful for legal learners & professionals  

---

## ✨ Features
- Retrieve the most relevant legal documents **before** answering  
- Works with **Indian Penal Code**, **Acts**, **Regulations**, etc.  
- Fully modular RAG architecture  
- Build your own FAISS-based legal knowledge base  
- Query through an easy-to-use Web UI  
- Add or update PDFs anytime  
- Jupyter notebooks for analysis & experimentation  

---

## 📂 Project Structure

```
bharat_law_gpt/
│
├── legal_docs/
│   └── pdf_files/          # Raw Indian legal documents used to build the vector DB
│
├── db/
│   └── faiss_store/        # Persistent FAISS vector index
│
├── src/                    # Core code: embeddings, retrieval, utilities, pipelines
│
├── notebooks/              # Experimentation & testing notebooks
│
├── build_db.py             # Script to ingest PDFs and build the FAISS database
├── app.py                  # Backend API for RAG queries
├── app_ui.py               # Frontend Interface (Streamlit/Gradio)
│
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🔍 How the RAG Pipeline Works

### **1️⃣ Document Ingestion**
PDFs inside `legal_docs/pdf_files/` are extracted, cleaned, and split into chunks.

### **2️⃣ Embedding & Vector Store**
- Each text chunk → embedding  
- Embeddings stored in **FAISS index** under `db/faiss_store`

### **3️⃣ User Query**
- Query is converted to an embedding  
- FAISS retrieves top relevant legal snippets

### **4️⃣ Answer Generation**
- Retrieved context + user query is passed to an LLM  
- Model generates a legally-grounded answer  
- Optionally returns citations & retrieved excerpts

---

## 🛠 Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/Rkgorai/bharat_law_gpt.git
cd bharat_law_gpt
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Add legal PDFs**
Place all legal documents into:
```
legal_docs/pdf_files/
```

### **4. Build the vector database**
```bash
python build_db.py
```

### **5. Run the application**

#### **Backend/API**
```bash
python app.py
```

#### **Frontend UI**
```bash
python app_ui.py
```

Then visit the shown URL (e.g., `http://localhost:8501`).

---

## 🧪 Example Usage

**User:**  
> "What does Section 420 IPC mean?"

**System flow:**  
- Query embedded → FAISS retrieves IPC Section 420 text  
- LLM processes retrieved content  
- Output includes explanation + legal source  

**Output:**  
> Section 420 of IPC deals with cheating and dishonestly inducing delivery of property…  
> **Source:** Indian Penal Code, Section 420  

---

## 📌 Why This Project Is Useful
- Legal knowledge is scattered across massive documents  
- Search engines are not tailored for legal phrasing  
- Generic LLMs hallucinate about law  
- Bharat Law GPT ensures answers are **based on real legal text**, improving trust & usability  

Ideal for:
- Law students  
- Legal researchers  
- Ordinary people seeking legal clarity  
- Developers exploring domain-specific RAG systems  

---

## 🧩 Future Enhancements
- Add Supreme Court / High Court judgments  
- Add case-law summarization  
- Improve chunking for long judgments  
- Switch to domain-tuned legal embeddings  
- Enable chat history and follow-up queries  
- Deploy on cloud (AWS/GCP/Azure/HuggingFace)

---

## 🚧 Limitations
⚠️ **This system is NOT a substitute for professional legal advice.**  
It is an educational and research tool.

---

## 👥 Contributing
Contributions are welcome!
- Add more legal documents  
- Improve RAG pipeline  
- Create better UI components  
- Submit issues or pull requests  

---

## ⭐ Support
If you like this project, please star the repo ⭐  
Your support encourages further development.
