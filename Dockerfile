# Use a lightweight Python image with a compatible Python version
FROM python:3.10-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port your app will run on
EXPOSE 8080

# Set an environment variable for the port, defaulting to 8080
ENV PORT=8080

# Run the Flask application
CMD ["python", "app.py"]
