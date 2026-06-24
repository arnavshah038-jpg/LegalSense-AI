import os
import json
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from pypdf import PdfReader
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

app = FastAPI(title="LegalTech AI Platform (Local AI + Qdrant)")

# 1. Connected to Qdrant Local Client
qdrant_client = QdrantClient(url="http://localhost:6333")

# 2. LOAD FROM LOCAL FOLDER
embedding_model = SentenceTransformer("./local_transformer_model") 

COLLECTION_NAME = "legal_contracts"

try:
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

class ContractMetadata(BaseModel):
    contract_title: str
    party_one: str
    party_two: str
    effective_date: str
    expiry_date: str
    jurisdiction: str

@app.get("/")
def read_root():
    return {"message": "LegalTech AI Platform with Qdrant is Live!"}

@app.post("/extract-and-store/", response_model=ContractMetadata)
async def extract_and_store(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        pdf_reader = PdfReader(file.file)
        contract_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                contract_text += text
        
        if not contract_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        prompt = f"""
        You are a legal assistant. Extract details from the contract text.
        Provide response ONLY as valid JSON matching this structure:
        {{
            "contract_title": "Title of contract",
            "party_one": "First party name",
            "party_two": "Second party name",
            "effective_date": "YYYY-MM-DD",
            "expiry_date": "YYYY-MM-DD",
            "jurisdiction": "Applicable law location"
        }}
        Return ONLY the raw JSON block without markdown wrappers like ```json.
        Contract Text:
        {contract_text[:8000]}
        """
        response = ollama.generate(model="llama3", prompt=prompt)
        response_text = response['response'].strip()

        # --- ROBUST JSON PARSING LOGIC START ---
        try:
            # Tarika 1: Seedhe load karne ki koshish karein
            cleaned_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Tarika 2: Agar LLM ne text ya ```json wrap kiya ho, toh regex se sirf {} nikalenge
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    cleaned_data = json.loads(json_match.group(0))
                except json.JSONDecodeError as je:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"LLM generated bad JSON structure. Error: {str(je)}. Raw response: {response_text[:200]}"
                    )
            else:
                # Tarika 3: Fallback data agar kuch na mile taaki code breakdown na ho
                cleaned_data = {
                    "contract_title": "Unknown Contract",
                    "party_one": "Not Extracted",
                    "party_two": "Not Extracted",
                    "effective_date": "None",
                    "expiry_date": "None",
                    "jurisdiction": "Unknown"
                }
        # --- ROBUST JSON PARSING LOGIC END ---

        # Vector conversion using the pre-downloaded local model
        text_vector = embedding_model.encode(contract_text[:5000]).tolist()
        point_id = int(os.urandom(4).hex(), 16)

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=text_vector,
                    payload={
                        "filename": file.filename,
                        "metadata": cleaned_data,
                        "text_snippet": contract_text[:1000]
                    }
                )
            ]
        )

        return ContractMetadata(**cleaned_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SearchQuery(BaseModel):
    query: str
    limit: int = Field(default=3, description="Kitne relevant results chahiye")

@app.post("/search/")
def search_contracts(search_data: SearchQuery):
    try:
        # 1. User ke sawal ko vector mein convert karein
        query_vector = embedding_model.encode(search_data.query).tolist()

        # 2. Qdrant mein query_points use karke search karein
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=search_data.limit
        )

        # 3. Results format karein
        formatted_results = []
        for hit in search_result.points:
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "filename": hit.payload.get("filename") if hit.payload else "Unknown",
                "metadata": hit.payload.get("metadata") if hit.payload else {},
                "text_snippet": hit.payload.get("text_snippet") if hit.payload else ""
            })

        return {"results": formatted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))