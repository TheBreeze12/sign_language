FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the build context (which will be the project root)
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/backend/

WORKDIR /app/backend

# Use Gunicorn in production ideally, but run app.py directly for development parity
CMD ["python", "app.py"]
