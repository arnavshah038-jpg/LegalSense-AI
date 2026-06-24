# LegalSense AI: Local Contract Analytics & Intelligence Platform
A secure, private, and 100% enterprise-ready local RAG (Retrieval-Augmented Generation) pipeline built to ingest legal contracts (PDFs), extract key structured metadata using Local LLMs, and provide semantic search capability via an isolated Vector Database.

System Architecture & Workflow
The platform is designed around data privacy, ensuring no legal documents or query parameters leave the local host environment.
1. Ingestion Pipeline: 
   - Raw PDF text parsed using `pypdf`.
   - Unstructured data is structured into schemas via a localized `Llama-3` instance controlled via `Ollama`.
   - Dense vector embeddings (384-dimensions) generated deterministically using a decoupled local `SentenceTransformer`.
2. Storage: High-performance payload indexing inside a localized `Qdrant` container.
3. Retrieval: High-speed hybrid spatial-similarity matching based on Cosine Distance metrics via Qdrant’s query points engine.

Core Stack Components
- Backend Engine: FastAPI (Asynchronous REST API Gateway)
- Frontend UI: Streamlit (Reactive Analytics Dashboard)
- Vector Store: Qdrant DB (Engineered via Docker Containerization)
- Local Inference: Ollama running `llama3:latest` (Deterministic JSON parsing structure)
- Embedding Model: SentenceTransformers (`all-MiniLM-L6-v2` compiled locally)

Prerequisites
Ensure your system has the following binaries available in the path environment:
- Python 3.10+
- Docker Engine / Desktop
- Ollama Client

Installation & Deployment Flow
Follow these execution steps sequentially in separate terminal instances to spin up the multi-layered ecosystem:

Step 1: Initialize Database Layer (Qdrant Container)
Execute the docker instance bound to local volumes to persist vector data profiles:

docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

Environment Setup & FastAPI Backend Deployment
1.Navigate to the project directory and instantiate the Python virtual environment:
(source venv/bin/activate)
2.Launch the backend application worker layer under high execution limits to accommodate local
model inference latency:
(uvicorn main:app --reload --timeout-keep-alive 1800 --workers 1)
{Verify API Gateway health status at: http://127.0.0.1:8000/docs}

Streamlit UI Dashboard Execution
3.Open a terminal instance, move into project directory, and activate environment context:
(source venv/bin/activate)
4.Boot the reactive frontend layout dashboard:
(streamlit run app.py) {Access Dashboard UI at: http://localhost:8501}
5.Monitoring Ecosystem Data
To structurally inspect the stored payloads, vector lengths, or semantic spaces inside the database, launch the interactive browser 
interface directly via:
{Qdrant DB Management UI: http://localhost:6333/dashboard}













