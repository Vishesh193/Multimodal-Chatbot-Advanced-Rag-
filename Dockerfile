# ==========================================
# Backend Dockerfile - InsureAI (FastAPI)
# ==========================================

# 1. Base Image
FROM python:3.11-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install System Dependencies (OCR, PDF Processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Project Code
COPY . .

# 7. Create Directories for Data/Logs
RUN mkdir -p logs chroma_db api_uploads extracted_images

# 8. Expose Backend Port
EXPOSE 8000

# 9. Run the Application
# Using uvicorn for high-performance async serving
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
