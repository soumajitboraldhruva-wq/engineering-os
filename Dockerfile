# Backend Dockerfile for Railway / Render / Fly.io / Cloud Run
FROM python:3.12-slim

WORKDIR /app

# Install system deps for C++ binary + Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ .

# Ensure the C++ binary exists (rebuild if needed)
RUN g++ -O2 -std=c++17 -o analyser_bin analyser.cpp

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Production: no reload, bind to 0.0.0.0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]