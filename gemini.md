# Gemini Workspace: InfiniteTalk Project

This document serves as a centralized knowledge base for the InfiniteTalk project. It outlines the project's architecture, key components, and operational procedures to guide development and maintenance.

## 1. Project Overview

InfiniteTalk is a real-time, low-latency talking avatar system. It generates a continuous video stream of an avatar speaking dialogue produced by an AI.

The core pipeline is as follows:
**LLM (e.g., OpenAI) → TTS (e.g., ElevenLabs) → Audio Chunks → InfiniteTalk Model → Video Segments → HLS Manifest → Frontend Player**

The system is designed to be a modular, event-driven streaming pipeline to ensure low latency.

## 2. Current Project Status

The project is currently a "modular monolith". The components for the pipeline are logically separated in the codebase but run within a single FastAPI process. This is a good starting point, but it is not scalable or robust.

The `Project_Architecture_Blueprint.md` file contains a detailed plan for evolving the project into a scalable, microservices-based architecture. The next steps should be focused on implementing the first phase of that blueprint.

## 3. Next Best Steps to MVP

The immediate goal is to achieve a Minimum Viable Pipeline (MVP) as defined in "Phase A" of the `Project_Architecture_Blueprint.md`. This involves the following steps:

1.  **Stabilize Core Processes**:
    *   Run the `ffmpeg` idle loop as a separate, managed process (e.g., in its own container or as a systemd service) to ensure it's always running.
    *   Add health checks to the main API to monitor the status of the `ffmpeg` process.

2.  **Decouple Components with a Message Bus**:
    *   Introduce Redis Streams as the message bus for communication between components.
    *   Extract the video generation logic into a separate Python worker process that consumes audio data from the Redis stream and produces video segments.
    *   Extract the TTS logic into a separate worker process that consumes text from the Redis stream and produces audio data.

3.  **Containerize the Application**:
    *   Create a `docker-compose.yml` file to define the services for the API, the new worker processes, and Redis.
    *   This will make the development environment consistent and prepare the project for production deployment.

4.  **Improve Observability**:
    *   Implement structured logging (e.g., JSON format) across all services, including a `session_id` to trace requests through the pipeline.
    *   Add basic Prometheus metrics to the API and workers to monitor key performance indicators like queue depth and processing latency.

## 4. Architecture

The target architecture, based on `Project_Architecture_Blueprint.md`, is a microservices-oriented approach using a message bus to decouple components.

*   **Control Plane / API (`src/app/main.py`)**: A FastAPI application that manages user sessions via WebSockets, handles user input, and orchestrates the pipeline by publishing messages to the message bus.
*   **Frontend (`web/`)**: A Next.js application with `hls.js` for video playback and WebSocket communication with the backend.
*   **LLM Streamer**: A worker that streams text from an LLM and breaks it into clauses.
*   **TTS Worker**: A worker that converts text clauses into audio data.
*   **Video Generator Worker**: A GPU-intensive worker that consumes audio and generates corresponding video segments using the InfiniteTalk model.
*   **HLS Writer (`src/app/hls_writer.py`)**: A component that uses FFmpeg to package video segments into an HLS stream (fMP4) and updates the manifest file.
*   **Message Bus**: Redis Streams to facilitate communication between workers.

## 5. Key Technologies

*   **Backend**: Python, FastAPI, Uvicorn
*   **Frontend**: Next.js, React, hls.js
*   **AI/ML**: PyTorch, InfiniteTalk model
*   **Streaming**: FFmpeg for HLS segmenting
*   **Dependencies**: Conda, Pip
*   **Containerization**: Docker

## 6. Setup & Running the Project (Current Monolithic Version)

### Backend

The backend is a FastAPI application.

1.  **Navigate to the source directory**:
    ```bash
    cd /home/canuk/projects/inifinitetalk-local/src
    ```
2.  **Install dependencies** (if not already done, based on `InfiniteTalk/README.md`):
    ```bash
    # Assuming a conda environment is active
    pip install -r ../InfiniteTalk/requirements.txt
    ```
3.  **Run the development server**:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
    ```

### Frontend

The frontend is a Next.js application.

1.  **Navigate to the web directory**:
    ```bash
    cd /home/canuk/projects/inifinitetalk-local/web
    ```
2.  **Install dependencies**:
    ```bash
    npm install
    ```
3.  **Run the development server**:
    ```bash
    npm run dev
    ```
    The frontend will be available at `http://localhost:3000`.
