# 🔍 Internal RFP Analyst — GenAI Agent Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that helps internal consulting teams instantly search past proposals, project outlines, RFP responses, and case studies. Built with **LangChain**, **LangGraph**, **ChromaDB**, and **Google Gemini**.

## 🏗️ Architecture

```
User Query → Streamlit UI → LangGraph ReAct Agent → Tool Selection
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
            RAG Retriever    Doc Summary     Project Compare
                    ↓               ↓               ↓
                ChromaDB      ChromaDB         ChromaDB
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                        Google Gemini LLM
                                    ↓
                    Cited Answer + Reasoning Trace
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Agent Framework** | LangChain + LangGraph (ReAct Pattern) |
| **Vector Database** | ChromaDB (Persistent) |
| **LLM** | Google Gemini 2.0 Flash (Free Tier) |
| **Embeddings** | Google text-embedding-004 |
| **PDF Processing** | PyMuPDF |
| **UI** | Streamlit |
| **Memory** | LangGraph MemorySaver |

## 🚀 Quick Start

### 1. Get a Free Google Gemini API Key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy the key

### 2. Setup Environment

```bash
# Clone and navigate
cd "EY Gen AI"

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key
copy .env.example .env
# Edit .env and paste your GOOGLE_API_KEY
```

### 3. Generate Mock Documents

```bash
python document_generator.py
```

This creates 10 realistic consulting documents in `data/documents/`.

### 4. Launch the App

```bash
streamlit run app.py
```

Then click **"🚀 Ingest Documents"** in the sidebar to index the PDFs.

## 🤖 Agent Tools

The AI agent has 4 specialized tools it can autonomously select:

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Semantic search over all documents via RAG |
| `get_document_summary` | Full summary of a specific document |
| `list_all_projects` | Overview of all indexed projects |
| `compare_projects` | Side-by-side comparison of multiple projects |

## 💬 Example Queries

- *"What tech stack did we use for the banking audit?"*
- *"Which projects used Azure services?"*
- *"Compare the healthcare and insurance projects"*
- *"What was the budget for the supply chain platform?"*
- *"List all projects with their timelines"*

## 📁 Project Structure

```
├── app.py                    # Streamlit chatbot UI
├── agent.py                  # LangGraph ReAct agent + tools
├── rag_engine.py             # RAG pipeline (ingest/embed/retrieve)
├── document_generator.py     # Mock PDF generator
├── config.py                 # Central configuration
├── requirements.txt          # Dependencies
├── .env.example              # API key template
├── .streamlit/config.toml    # UI theme
├── data/documents/           # PDF documents
└── vectorstore/              # ChromaDB persistent storage
```

## 📜 License

This project is for educational and portfolio demonstration purposes.
