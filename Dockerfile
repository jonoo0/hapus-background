# app/Dockerfile
FROM python:3.9.21-slim-bullseye

WORKDIR /home/app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/jonoo0/hapus-background.git .

# Create a virtual environment named remove_bg
RUN python3 -m venv /home/app/remove_bg

# Activate the virtual environment and install PyTorch, Torchvision, Torchaudio, and other dependencies
RUN /bin/bash -c "source /home/app/remove_bg/bin/activate && \
    pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt"

# Create the output_images directory
RUN mkdir /home/app/output_images

EXPOSE 8501

# Add a healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Use the virtual environment for running the app
ENTRYPOINT ["/bin/bash", "-c", "source /home/app/remove_bg/bin/activate && streamlit run app.py --server.port=8501"]
