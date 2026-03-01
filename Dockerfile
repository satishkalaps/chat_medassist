FROM python:3.11-slim

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=off" pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV N_GPU_LAYERS=0
ENV PORT=8000

# Start the server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
