# ==========================================
# Unified Dockerfile for Hugging Face Spaces
# ==========================================

# --- Stage 1: Build React Frontend ---
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# --- Stage 2: Final AI Backend ---
FROM python:3.11-slim

# Install system dependencies for AI & OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential tesseract-ocr poppler-utils libmagic1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Copy built frontend assets from Stage 1 to a folder FastAPI can serve
COPY --from=frontend-build /frontend/dist ./frontend_dist

# Create necessary directories
RUN mkdir -p logs chroma_db api_uploads extracted_images && chmod 777 -R /app

# Expose the port Hugging Face expects
EXPOSE 7860

# Command to start the app
# We tell Uvicorn to run on 7860 as per HF requirements
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
