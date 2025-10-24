FROM python:3.11-slim

# Install system deps for wkhtmltopdf and build tools for tpot dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    git \
    python3-dev \
    libssl-dev \
    libffi-dev \
    libxrender1 \
    libfontconfig1 \
    libx11-6 \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install wkhtmltopdf (static)
RUN wget -q https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.buster_amd64.deb -O /tmp/wkhtmltox.deb \
    && apt-get update && apt-get install -y /tmp/wkhtmltox.deb || true \
    && rm -f /tmp/wkhtmltox.deb

WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "main.py"]
