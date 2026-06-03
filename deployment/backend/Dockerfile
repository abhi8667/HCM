FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for compiling some packages if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files and dynamic database
COPY . .

# Expose backend port
EXPOSE 5000

# Run with Gunicorn (non-root worker configuration recommended for production)
CMD ["gunicorn", "--workers", "1", "--threads", "4", "-b", "0.0.0.0:5000", "app:app"]
