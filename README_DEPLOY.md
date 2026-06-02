# 🐳 Bharat Law GPT - Containerized Deployment Guide

This guide explains how to build, run, and maintain the decoupled, containerized **Bharat Law GPT** architecture.

---

## 🏗️ Prerequisites

Ensure you have the following installed on your host system:
1. **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
2. **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)

---

## ⚡ Step-by-Step Launch Instructions

### **1. Set up your Environment Variable**
Create (or edit) the `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY="your-groq-api-key-here"
```

### **2. Build and Start the Containers**
Launch both services in the background (detached mode):
```bash
docker-compose up --build -d
```
This command automatically:
- Builds the backend API server with full AI/ML models (`faster-whisper`, `sentence-transformers`, `faiss`).
- Builds the frontend Streamlit server as a clean, ultra-lightweight client shell.
- Sets up decoupled inter-container networking (`http://backend:8000`).
- Bind-mounts your host `db/` and `legal_docs/` folders.

### **3. Verify the Services are Running**
Check container logs or status:
```bash
docker-compose ps
```
- **Streamlit Frontend** is accessible at: `http://localhost:8501`
- **FastAPI Backend** is accessible at: `http://localhost:8000`

---

## 📂 Managing Documents and Database Indexing

The `db/` and `legal_docs/` folders are **bind-mounted** into the backend container. This means any database index files and PDFs reside safely on your host machine and persist across container restarts.

### **Adding New PDF Documents**
1. Simply copy your PDF files (e.g. acts, contracts, rulings) on your host machine to:
   ```text
   legal_docs/pdf_files/
   ```
2. Re-trigger the hybrid indexing script **inside the running backend container**:
   ```bash
   docker-compose exec backend python build_db.py
   ```
3. Once completed, the new indexes are immediately saved to the host `db/faiss_store/` and loaded by the backend without rebuilding docker images!

---

## 🛠️ Orchestration Commands Reference

| Operation | Command |
| :--- | :--- |
| **Start Services** | `docker-compose up -d` |
| **Stop Services** | `docker-compose down` |
| **Rebuild & Start** | `docker-compose up --build -d` |
| **Check Logs (all)** | `docker-compose logs -f` |
| **Check Logs (backend)** | `docker-compose logs -f backend` |
| **Check Logs (frontend)** | `docker-compose logs -f frontend` |
| **Re-index Documents** | `docker-compose exec backend python build_db.py` |

---

## 🛡️ Production & Security Safeguards
- **PII Anonymization:** Personally Identifiable Information is processed and masked on the server *before* sending requests to Groq, protecting user privacy.
- **Prompt Injection Defense:** Instantly intercepts and blocks off-topic queries or override prompts before hitting RAG or wasting API tokens.
