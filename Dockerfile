# Use Python 3.12 slim as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (similar to your local venv setup)
RUN pip install --no-cache-dir -r requirements.txt

# Install python-dotenv separately as it seems to be missing from requirements list but is imported
RUN pip install --no-cache-dir python-dotenv

# Copy the application code
COPY . .

# Create a directory for templates if it doesn't exist
RUN mkdir -p templates

# Expose the port the app runs on (matches your local 127.0.0.1:5000)
EXPOSE 5000

# Set environment variables
ENV TOGETHER_API_KEY="YOUR_API" \
    CLOUDINARY_CLOUD_NAME="YOUR_API" \
    CLOUDINARY_API_KEY="YOUR_API" \
    CLOUDINARY_API_SECRET="YOUR_API"

# Command to run the Flask app directly (similar to how you run it locally)
# We use 0.0.0.0 instead of 127.0.0.1 to make it accessible from outside the container
CMD ["python", "app.py"]
