FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirement.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Copy application code
COPY backend/ /app/

# Fix permissions
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:////app/data/customer_support.db

EXPOSE 8000

# Initialize database and start application
CMD ["sh", "-c", "python init_db.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
