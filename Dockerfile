# Use the official CUDA image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /mab_recsys

# Install dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 curl bzip2 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add Conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create a Conda environment with Python 3.8
RUN conda create -n py38 python=3.8 -y && \
    conda clean --all --yes

# Activate the Conda environment
SHELL ["conda", "run", "-n", "py38", "/bin/bash", "-c"]

# Install pip and other Python dependencies
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container
COPY src ./src

# Default to running in the Conda environment
CMD ["conda", "run", "-n", "py38", "python", "src/main.py"]