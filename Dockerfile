FROM python:3.10-slim

WORKDIR /app


# Install system dependencies including curl
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


