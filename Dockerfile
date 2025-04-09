FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (needed for PaddleOCR + OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install paddleocr
RUN pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Expose port for Railway
EXPOSE 8080

# Start Flask app via Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
