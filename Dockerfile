# Use a lightweight Python base image
FROM python:3.9-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port (adjusted to detect dynamic port if needed)
EXPOSE 9090

# Set environment variable for the port (optional, depending on deployment setup)
ENV PORT=9090

# Run the application
CMD ["python", "app.py"]
