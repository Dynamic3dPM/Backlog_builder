# Hugging Face Local LLM Processing Service
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (PyTorch already included in base image)
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True)"

# Create directory for models
RUN mkdir -p /app/models/summarization

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Expose port
EXPOSE 8002

# Run the application
CMD ["uvicorn", "huggingface-local:app", "--host", "0.0.0.0", "--port", "8002"]
