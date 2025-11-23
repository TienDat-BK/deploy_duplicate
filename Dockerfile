# Base Python
FROM python:3.11-slim

# Install system packages needed for C++ build
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy dependency list first
COPY requirements.txt .

# Install Python dependencies + build dependencies
RUN pip install --upgrade pip setuptools wheel pybind11 \
    && pip install -r requirements.txt

# Copy entire project
COPY . .

# Build C++ extension from setup.py
RUN python setup.py build_ext --inplace

# Expose port for web app (Gradio, Flask...)
EXPOSE 10000

# Run visualize.py when container starts
CMD ["python", "visualize.py"]
