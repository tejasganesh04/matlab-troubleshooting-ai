# MATLAB Troubleshooting Assistant

A production-ready AI-powered assistant designed to debug, analyze, and solve MATLAB and Simulink errors using LangGraph, Gemini 2.5 Flash, Gemini Vision, and FAISS.

## Features
- **Intent Recognition**: Routes queries to General LLM, RAG, Vision, or Web Search.
- **Multi-modal Vision**: Extracts and analyzes errors or Simulink blocks from screenshots.
- **Corrective RAG (CRAG)**: Rewrites query and triggers web search if retrieved documentation context is insufficient.
- **Self-RAG (Self-Reflective Validation)**: Validates response groundedness and completeness.

## Project Structure
The repository is modularized as follows:
- `app.py`: Streamlit frontend.
- `agents/`: Custom LLM reasoning wrappers.
- `graph/`: LangGraph workflow nodes, edges, state, and builder.
- `rag/`: Loader, splitter, embedding, and FAISS vector indexer.
- `prompts/`: System/user prompt templates.
- `config/`: settings and variable loader.
- `knowledge_base/pdfs/`: Reference documentation PDFs.
- `faiss_index/`: Compiled FAISS search index storage.

## Setup Instructions
1. Clone the repository and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your keys in `.env`:
   ```
   GEMINI_API_KEY=your_key
   ```
3. Place MATLAB manuals/PDFs into `knowledge_base/pdfs/`.
4. Index PDF content:
   ```bash
   python rag/indexer.py
   ```
5. Run the Streamlit web interface:
   ```bash
   streamlit run app.py
   ```
