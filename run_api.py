import uvicorn

if __name__ == "__main__":
    print("=========================================")
    print("🚀 Starting Insurance RAG API Server...")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=========================================")
    
    # Run the FastAPI app
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
