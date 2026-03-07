"""
api/main.py — FastAPI Server for Insurance RAG
================================================
This is the bridge between the Python backend and the React frontend.
It provides REST API endpoints for all Insurance RAG features.
"""

import os
import shutil
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from insurance import InsuranceRAG
from api.models import (
    QueryRequest, 
    ComparePoliciesRequest, 
    ChecklistRequest, 
    ExclusionRequest,
    IsExcludedRequest
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── 1. Create FastAPI App ─────────────────────────────────────────────────
app = FastAPI(
    title="Insurance RAG API",
    description="Backend API for the Multimodal Insurance Claim Assistant",
    version="1.0.0"
)

# ── 2. Configure CORS (Cross-Origin Resource Sharing) ─────────────────────
# This allows your React frontend (e.g. localhost:3000) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 3. Initialize the Global RAG System ───────────────────────────────────
# We initialized this globally so it only loads into memory ONCE on startup
rag: InsuranceRAG = None # type: ignore

@app.on_event("startup")
async def startup_event():
    global rag
    logger.info("🚀 Starting FastAPI Server. Initializing Insurance RAG...")
    try:
        rag = InsuranceRAG()
        logger.info("✅ Insurance RAG loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load Insurance RAG: {e}")
        # In a real app we might raise an error here to prevent starting a broken server

# Helper to provide the RAG instance
def get_rag() -> InsuranceRAG:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system is still loading or failed to load")
    return rag

# ── Temporary upload directory ──────────────────────────────────────────
UPLOAD_DIR = "./api_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── 4. API Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Check if the server is running."""
    return {"status": "healthy", "service": "Insurance RAG API"}

@app.get("/status")
async def system_status(rag_sys: InsuranceRAG = Depends(get_rag)):
    """Get the status of the RAG system and LLMs."""
    return rag_sys.get_status()

# ── Policy Management Endpoints ──

@app.post("/policies/ingest")
async def ingest_policy(
    file: UploadFile = File(...),
    policy_name: str = Form(...),
    insurer: str = Form(...),
    policy_type: str = Form("health"),
    policy_number: str = Form(""),
    sum_insured: str = Form(""),
    premium: str = Form(""),
    holder_name: str = Form(""),
    tags_string: str = Form(""),
    rag_sys: InsuranceRAG = Depends(get_rag)
):
    """
    Upload and ingest an insurance policy PDF.
    
    This endpoint uses multipart/form-data so you can upload a file and send text data together.
    """
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Parse tags
        tags = [t.strip() for t in tags_string.split(",")] if tags_string else []
        
        # Ingest
        policy_id = rag_sys.ingest_policy(
            pdf_path=temp_path,
            policy_name=policy_name,
            insurer=insurer,
            policy_type=policy_type,
            policy_number=policy_number,
            sum_insured=sum_insured,
            premium=premium,
            holder_name=holder_name,
            tags=tags
        )
        
        return {
            "success": True, 
            "message": f"Policy '{policy_name}' ingested successfully", 
            "policy_id": policy_id
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/policies")
async def list_policies(policy_type: str = None, rag_sys: InsuranceRAG = Depends(get_rag)):
    """List all registered insurance policies."""
    policies = rag_sys.list_policies(policy_type)
    return {"policies": policies}

# ── Core Insurance Features ──

@app.post("/query")
async def query_assistant(req: QueryRequest, rag_sys: InsuranceRAG = Depends(get_rag)):
    """Ask a general insurance question."""
    try:
        result = rag_sys.query(
            user_query=req.query,
            policy_name=req.policy_name,
            language=req.language,
            include_images=req.include_images,
            max_tokens=req.max_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_insurance_policies(req: ComparePoliciesRequest, rag_sys: InsuranceRAG = Depends(get_rag)):
    """Compare two policies side-by-side."""
    try:
        result = rag_sys.compare_policies(
            policy_name_a=req.policy_name_a,
            policy_name_b=req.policy_name_b,
            query=req.query,
            language=req.language
        )
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/claim-checklist")
async def generate_claim_checklist(req: ChecklistRequest, rag_sys: InsuranceRAG = Depends(get_rag)):
    """Generate a step-by-step claim checklist."""
    try:
        result = rag_sys.get_claim_checklist(
            user_query=req.query,
            policy_name=req.policy_name,
            language=req.language
        )
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/exclusions")
async def fetch_exclusions(req: ExclusionRequest, rag_sys: InsuranceRAG = Depends(get_rag)):
    """Get all exclusion clauses highlighted."""
    try:
        result = rag_sys.find_exclusions(
            user_query=req.query,
            policy_name=req.policy_name,
            language=req.language
        )
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/is-excluded")
async def check_item_exclusion(req: IsExcludedRequest, rag_sys: InsuranceRAG = Depends(get_rag)):
    """Quick yes/no lookup if an item is excluded."""
    try:
        return rag_sys.is_excluded(item=req.item, policy_name=req.policy_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Computer Vision Endpoints ──

@app.post("/analyse-bill")
async def analyse_medical_bill(file: UploadFile = File(...), rag_sys: InsuranceRAG = Depends(get_rag)):
    """Upload a hospital bill or discharge summary image to extract structured JSON data."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file supplied")
        
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = rag_sys.analyse_bill(temp_path)
        if "error" in result.get("extracted_data", {}):
            raise HTTPException(status_code=500, detail=result["extracted_data"]["error"])
            
        return result
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # When you run tests locally you can just launch this file `python api/main.py`
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
