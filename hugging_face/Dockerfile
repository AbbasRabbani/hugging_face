# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY requirements.txt .

# Update the package list and install system dependencies
RUN apt-get update && apt-get install -y gcc python3-dev

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the Hugging Face token as an environment variable
ENV HF_TOKEN=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

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5010

# Run app.py when the container launches
#CMD ["python", "app.py"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5010"]
