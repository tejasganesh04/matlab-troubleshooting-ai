# MATLAB Troubleshooting AI

Project owner: **Tejas Ganesh**.

An AI assistant for investigating MATLAB errors and Simulink models, with screenshot analysis, local PDF retrieval, and web search.

[Repository](https://github.com/tejasganesh04/matlab-troubleshooting-ai)

## What it does

- Routes questions between general answers, PDF retrieval, screenshot analysis, and web search.
- Uses Gemini for text and image analysis, FAISS for documentation search, and Tavily for web context.
- Rewrites unsuccessful retrieval queries and runs a bounded answer validation loop.
- Displays formatted answers and keeps the latest answer available across Streamlit reruns, with a Markdown download.

## Run locally

Use Python 3.11 or 3.12 in an isolated environment:

```bash
git clone https://github.com/tejasganesh04/matlab-troubleshooting-ai.git
cd matlab-troubleshooting-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows, activate with `.venv\Scripts\activate`.

Fill in all four API/model values in `.env`. `GEMINI_API_KEY` is your Google API key; `TAVILY_API_KEY` enables web search. Choose a Gemini generation model and Google embedding model available to your account for `GEMINI_MODEL` and `EMBEDDING_MODEL`. Model availability depends on the provider and your account. `LOG_LEVEL` defaults to `INFO`.

```bash
streamlit run app.py
```

Enter a question and optionally attach a PNG or JPEG screenshot. Click **Generate Answer**. The model can make mistakes; check proposed MATLAB commands and model changes before using them.

## Add your own documentation

Place PDFs you are allowed to use in `knowledge_base/pdfs/`, then run this command from the repository root:

```bash
python -m rag.indexer
```

Restart Streamlit after rebuilding an index. Rebuild whenever you change the embedding model. Without an index, retrieval returns empty context and the workflow can fall back to web search. Only load FAISS indexes you generated yourself: the local index loader deserializes pickle data.

## Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `agents/` | Classification, vision, retrieval, search, generation, and grading |
| `graph/` | LangGraph state, nodes, and bounded retry routing |
| `rag/` | PDF loading, chunking, embeddings, and FAISS indexing |
| `prompts/` | Editable agent prompts |
| `config/settings.py` | Environment configuration |
| `.env.example` | Required configuration template |

## Troubleshooting

- **Missing environment variable:** populate every required value in `.env` and restart the app.
- **Empty retrieval:** add PDFs and run `python -m rag.indexer` from the root.
- **API failures:** check keys, model access, quotas, and network connectivity.
- **Dependency install failures:** use a fresh Python 3.11 or 3.12 virtual environment.

Questions, images, and retrieved context may be sent to Google; search queries are sent to Tavily. Keep confidential material out of requests unless you have permission to share it. Secrets, uploaded files, PDFs, and generated indexes are excluded from Git.
