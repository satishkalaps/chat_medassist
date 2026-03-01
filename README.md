# Medical RAG Assistant

AI-powered medical question-answering web application built with **FastAPI**, **LangChain**, **ChromaDB**, and **Mistral-7B**. Uses Retrieval-Augmented Generation (RAG) to answer clinical, diagnostic, and treatment questions based on the Merck Medical Manual.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)

---

## Project Structure

```
medical-rag-app/
├── main.py                 # FastAPI server & API routes
├── rag_pipeline.py         # RAG pipeline (PDF → Chunks → Embeddings → LLM)
├── static/
│   └── index.html          # Frontend UI (single-file HTML/CSS/JS)
├── data/
│   └── medical_diagnosis_manual.pdf   # ← Place your PDF here
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## How It Works

1. **PDF Loading** — PyMuPDFLoader reads the medical manual
2. **Chunking** — RecursiveCharacterTextSplitter splits into 512-token chunks (128 overlap)
3. **Embedding** — BAAI/bge-base-en-v1.5 generates vector embeddings
4. **Vector Store** — ChromaDB stores and indexes the embeddings
5. **Retrieval** — Top 3 most similar chunks are retrieved per query
6. **Generation** — Mistral-7B-Instruct generates an answer using retrieved context

---

## Local Setup

### Prerequisites
- Python 3.10 or 3.11
- ~8 GB free RAM (for Mistral-7B model)
- ~6 GB free disk space (model download)

### Step 1: Clone & Set Up

```bash
git clone <your-repo-url>
cd medical-rag-app
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Add the PDF

Place `medical_diagnosis_manual.pdf` in the `data/` folder:

```
data/medical_diagnosis_manual.pdf
```

### Step 3: Run the App

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first run, the app will:
1. Download the embedding model (~440 MB)
2. Download Mistral-7B GGUF (~5.5 GB)
3. Build the ChromaDB vector database from the PDF

Subsequent runs will be much faster as everything is cached.

### Step 4: Open the App

Visit **http://localhost:8000** in your browser.

---

## Deploy to Railway

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

> **Important**: Make sure `medical_diagnosis_manual.pdf` is in the `data/` folder and committed to the repo. If the PDF is too large for GitHub (>100MB), use [Git LFS](https://git-lfs.com/).

### Step 2: Deploy on Railway

1. Go to [railway.app](https://railway.app) and sign in with GitHub
2. Click **"New Project"** → **"Deploy from GitHub Repo"**
3. Select your repository
4. Railway will auto-detect the Dockerfile and start building
5. Once deployed, click **"Generate Domain"** to get your public URL

### Environment Variables (optional)

In Railway's dashboard, go to **Variables** and add:

| Variable        | Value | Description                     |
|-----------------|-------|---------------------------------|
| `N_GPU_LAYERS`  | `0`   | Set > 0 if GPU available        |
| `PORT`          | `8000`| Railway sets this automatically  |

### Notes on Railway

- **Free tier**: $5/month credit (enough for personal use)
- **Build time**: First deploy takes 10-15 min (downloads models)
- **RAM**: Choose a plan with at least 8 GB RAM for Mistral-7B
- The vector database is rebuilt on each new deploy (takes a few minutes)

---

## API Reference

### `POST /api/ask`

Ask a medical question.

**Request:**
```json
{
  "question": "What are the symptoms of pulmonary embolism?",
  "max_tokens": 1024,
  "temperature": 0.0
}
```

**Response:**
```json
{
  "answer": "The symptoms of pulmonary embolism include...",
  "sources": ["chunk text 1...", "chunk text 2...", "chunk text 3..."],
  "response_time": 4.32
}
```

### `GET /api/health`

Health check endpoint.

---

## Configuration

Key settings can be modified in `rag_pipeline.py`:

| Setting              | Default                          | Description                      |
|----------------------|----------------------------------|----------------------------------|
| `CHUNK_SIZE`         | 512                              | Token size per chunk             |
| `CHUNK_OVERLAP`      | 128                              | Overlap between chunks           |
| `TOP_K`              | 3                                | Number of retrieved chunks       |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5`       | Sentence transformer model       |
| `MODEL_REPO`         | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | HuggingFace model repo |
| `MODEL_FILE`         | `mistral-7b-instruct-v0.2.Q6_K.gguf`     | Specific GGUF file      |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Use a smaller GGUF quantization (e.g., `Q4_K_M` instead of `Q6_K`) |
| Slow first startup | Normal — models are downloading. Subsequent starts are fast |
| PDF not found error | Ensure `medical_diagnosis_manual.pdf` is in the `data/` folder |
| Railway deploy fails | Check logs; ensure Dockerfile builds. You may need a plan with more RAM |

---

## Credits

- **Medical Data**: Merck Manual (4,000+ pages, 23 sections)
- **LLM**: Mistral-7B-Instruct-v0.2 (via llama-cpp-python)
- **Embeddings**: BAAI/bge-base-en-v1.5
- **Framework**: FastAPI + LangChain + ChromaDB

---

*Built as part of the UT Austin PGP AIML — NLP with Generative AI module.*
