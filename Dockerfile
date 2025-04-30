# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app
# Install system libraries needed for OpenCV etc
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Copy your project files into the container
COPY . .

# Install required packages
RUN pip install --upgrade pip
RUN pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt
RUN python download_ckpts.py

# Expose the port (your FastAPI runs on 8002)a
EXPOSE 8002

# Run the application
CMD ["uvicorn", "fapi:app", "--host", "0.0.0.0", "--port", "8002"]

