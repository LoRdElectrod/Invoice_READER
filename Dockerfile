FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies (important for PaddleOCR and OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install paddleocr
RUN pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
RUN pip install setuptools 

# Expose the port Flask runs on
EXPOSE 8080

# Start the app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
