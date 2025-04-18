FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Copy application code
COPY backend/ /app/

# Copy .env file for environment variables if needed
COPY .env /app/.env

# Fix permissions
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
# Using PostgreSQL in docker-compose, so this is overridden there
ENV DATABASE_URL=sqlite:////app/data/customer_support.db

EXPOSE 8000

# Initialize database and start application
CMD ["sh", "-c", "python init_db.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
