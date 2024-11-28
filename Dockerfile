# Use the official Python 3.11.7 image
FROM python:3.11.7

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to the new user's home
WORKDIR /home

# Copy the requirements file into the container
COPY requirements.txt /home/

# Install the dependencies as the root user
RUN pip install --no-cache-dir -r /home/requirements.txt

# Default command is to open a bash shell
CMD ["bash"]