# InfiniteTalk Project Cheat Sheet

This document contains common commands for developing and running the InfiniteTalk project.

## 🐳 Docker (Recommended for Backend)

Use these commands to run the Backend API, Redis, and Workers.

### **Start Everything**
Builds images and starts containers in the background.
```bash
docker-compose up --build -d
```

### **Manage Containers**
- **Stop containers**:
  ```bash
  docker-compose stop
  ```
- **Stop and remove containers** (cleans up network):
  ```bash
  docker-compose down
  ```
- **View logs** (follow output):
  ```bash
  docker-compose logs -f
  ```
- **View logs for specific service** (e.g., api, video-worker):
  ```bash
  docker-compose logs -f api
  docker-compose logs -f video-worker
  ```

### **Rebuild Specific Service**
If you installed new Python dependencies:
```bash
docker-compose build api
docker-compose up -d api
```

---

## 🌐 Frontend (Next.js)

The frontend is currently **not** in Docker. Run it locally on your machine.

### **Setup (First Time)**
```bash
cd web
npm install
```

### **Start Development Server**
Runs at [http://localhost:3001](http://localhost:3001)
```bash
cd web
npm run dev
```

---

## 🐍 Local Backend (No Docker)

If you prefer running Python directly on your host machine.

### **Setup Environment**
```bash
# If using conda:
conda create -n infinitetalk python=3.12
conda activate infinitetalk

# If using venv (already in the repo):
source venv/bin/activate  # On Linux/WSL
.\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r InfiniteTalk/requirements.txt
```

### **Start Redis**
Redis is required for message passing.
```bash
# If you have redis installed on Windows/WSL:
redis-server
# OR run ONLY redis via docker:
docker-compose up -d redis
```

### **Start API**
Runs at [http://localhost:8000](http://localhost:8000)
```bash
# From the src directory
cd src
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### **Start Worker**
```bash
# From the project root
python InfiniteTalk/workers/video/video_worker.py
```

---

## 🛠️ Troubleshooting

- **Check running containers**:
  ```bash
  docker ps
  ```
- **Check Redis connection**:
  Inside the container:
  ```bash
  docker exec -it infinitetalk_redis redis-cli ping
  ```
- **Prune Docker System** (If disk is full or cache is stale):
  ```bash
  docker system prune -a
  ```
- **Container Config Errors / Missing Variables**:
  If you see errors about missing keys (e.g., `RUNPOD_API_KEY`) or empty strings in `docker-compose config`, ensure your environment variables are loaded:
  ```bash
  # Explicitly load .env before running docker-compose
  set -a && source .env && set +a
  docker-compose up -d
  ```

---

## 🚀 RunPod Cached Models

Accelerate cold starts by using cached models.
- **Reference Document**: [cache_models_runpod.md](file:///InfiniteTalk/assets/cache_models_runpod.md)
- **Utility**: Use `src/app/runpod_utils.py` to find models programmatically.
