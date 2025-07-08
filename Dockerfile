# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire app
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI app with uvicorn on port 7860 (mandatory for Hugging Face)
CMD ["uvicorn", "newapi:app", "--host", "0.0.0.0", "--port", "7860"]
