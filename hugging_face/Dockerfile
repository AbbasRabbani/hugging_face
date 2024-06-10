# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the requirements file
COPY requirements.txt .

# Update the package list and install system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the Hugging Face token as an environment variable
ENV HF_TOKEN=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

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Accept a build argument for the port
ARG PORT

# Expose the port the app will run on
EXPOSE ${PORT}

# Command to run the application with the specified port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

