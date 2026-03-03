"""
RAG Pipeline Module (Groq API version)
Refactored from NLP_RAG_Project_Notebook.ipynb

This module handles:
- PDF loading and chunking
- Embedding generation
- Vector store creation and retrieval
- LLM-based question answering with RAG via Groq API
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "medical_db")
PDF_PATH = os.path.join(DATA_DIR, "medical_diagnosis_manual.pdf")

# Groq model (same Mistral model family, but run via API)
GROQ_MODEL = "llama-3.3-70b-versatile"

# Chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
ENCODING_NAME = "cl100k_base"

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval settings
TOP_K = 3

# Prompt templates
QNA_SYSTEM_MESSAGE = """
You are an assistant whose work is to review the report and provide the appropriate answers from the context.
User input will have the context required by you to answer user questions.
This context will begin with the token: ###Context.
The context contains references to specific portions of a document relevant to the user query.

User questions will begin with the token: ###Question.

Please answer only using the context provided in the input. Do not mention context in your final answer.

If the answer is not found in the context, respond "Sorry, I was not able to find the information".
"""

QNA_USER_MESSAGE_TEMPLATE = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""


class RAGPipeline:
    """
    Encapsulates the full RAG pipeline:
    PDF → Chunks → Embeddings → Vector Store → Retriever → Groq LLM Response
    """

    def __init__(self):
        self.groq_client = None
        self.vectorstore = None
        self.retriever = None
        self.embedding_model = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self):
        """Load embedding model, build/load vector store, and set up Groq client."""
        if self._initialized:
            return

        print("[1/3] Loading embedding model...", flush=True)
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

        print("[2/3] Building / loading vector store...", flush=True)
        self._build_vectorstore()

        print("[3/3] Connecting to Groq API...", flush=True)
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
            print("  → Groq API key found and client created.", flush=True)
        else:
            print("  → GROQ_API_KEY not found at startup, will check at query time.", flush=True)


        self._initialized = True
        print("✓ RAG Pipeline ready!", flush=True)

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    def _build_vectorstore(self):
        """Create the Chroma vector DB from the PDF, or load existing one."""
        if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
            print("  → Loading existing vector database...", flush=True)
            self.vectorstore = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=self.embedding_model,
            )
        else:
            print("  → Creating vector database from PDF (first-time setup)...", flush=True)
            if not os.path.exists(PDF_PATH):
                raise FileNotFoundError(
                    f"PDF not found at {PDF_PATH}. "
                    "Please place 'medical_diagnosis_manual.pdf' in the data/ folder."
                )

            # Load & chunk
            loader = PyMuPDFLoader(PDF_PATH)
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=ENCODING_NAME,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            document_chunks = loader.load_and_split(text_splitter)
            print(f"  → PDF split into {len(document_chunks)} chunks. Creating embeddings...", flush=True)
            # Create vector store
            os.makedirs(VECTOR_DB_DIR, exist_ok=True)
            self.vectorstore = Chroma.from_documents(
                document_chunks,
                self.embedding_model,
                persist_directory=VECTOR_DB_DIR,
            )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        user_input: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        """
        Run the full RAG pipeline for a user question.

        Returns:
            dict with keys: 'answer', 'sources' (list of retrieved chunk texts)
        """
        if not self._initialized:
            # Lazy-load Groq client if not set during startup
            if not self.groq_client:
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    return {"answer": "Error: GROQ_API_KEY is not set.", "sources": []}
                self.groq_client = Groq(api_key=api_key)
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # 1. Retrieve relevant document chunks
        relevant_docs = self.retriever.get_relevant_documents(user_input)
        context_list = [doc.page_content for doc in relevant_docs]
        context_for_query = ". ".join(context_list)

        # 2. Build the prompt
        user_message = QNA_USER_MESSAGE_TEMPLATE.replace("{context}", context_for_query)
        user_message = user_message.replace("{question}", user_input)

        # 3. Generate response via Groq API
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": QNA_SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message},
                ],
                model=GROQ_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            answer = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Sorry, I encountered the following error:\n{e}"

        return {
            "answer": answer,
            "sources": context_list,
        }


# ---------------------------------------------------------------------------
# Singleton instance for the app to use
# ---------------------------------------------------------------------------
pipeline = RAGPipeline()
