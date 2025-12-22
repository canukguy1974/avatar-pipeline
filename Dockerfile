# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory into the container
COPY . .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --timeout=100 -r InfiniteTalk/requirements.base.txt
RUN apt-get update && apt-get install -y ffmpeg


# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables (can be overridden by docker-compose)
ENV PYTHONUNBUFFERED=1

# The command to run the API server will be specified in docker-compose.yml
