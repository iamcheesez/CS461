# Dockerfile for What DAWG!? Dog Breed Classifier
# Multi-stage build for smaller image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Install runtime dependencies for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app.py wsgi.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/ ./models/
COPY breed_info_en.json breed_info_th.json ./

# Create necessary directories
RUN mkdir -p uploads feedback_data

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# Run with gunicorn
CMD ["gunicorn", "wsgi:app", "--workers", "2", "--timeout", "120", "--bind", "0.0.0.0:5000"]
