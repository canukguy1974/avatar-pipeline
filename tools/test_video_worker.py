import redis
import json
import uuid
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='/home/canuk/projects/inifinitetalk-local/.env')

# Redis connection details
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
AUDIO_STREAM = "audio_windows"

# --- Job Definition ---
# This is a sample job that mimics what the RollingWav component would produce.
# You can modify these values to test different scenarios.

# 1. Define the paths to your conditional inputs
# The base image for the talking head
# COND_VIDEO_PATH = os.getenv("INFINITETALK_BASE_IMAGE_URL") # DEBUG: This is being overridden by the environment
COND_VIDEO_PATH = "/home/canuk/projects/inifinitetalk-local/src/static/me_desk.png"
# A sample audio file to drive the animation
COND_AUDIO_PATH = "/home/canuk/projects/inifinitetalk-local/audio/audio_0000_0_1800.wav"

# 2. Check if the input files exist
if not COND_VIDEO_PATH or not os.path.exists(COND_VIDEO_PATH):
    print(f"Error: Conditional video file not found at '{COND_VIDEO_PATH}'. Please check the INFINITETALK_BASE_IMAGE_URL in your .env file.")
    exit(1)
if not os.path.exists(COND_AUDIO_PATH):
    print(f"Error: Conditional audio file not found at '{COND_AUDIO_PATH}'.")
    exit(1)

# 3. Define the generation parameters
# These arguments are passed to the generate_infinitetalk.py script.
# You can adjust them to control the output.
GENERATION_ARGS = {
    "size": "infinitetalk-480",
    "motion_frame": 9,
    "frame_num": 81,
    "sample_shift": 7,
    "sd_steps": 40,
    "text_guide_scale": 5.0,
    "audio_guide_scale": 4.0,
    "seed": 42,
    "color_correction_strength": 1.0,
    "task": "infinitetalk-14B",
    "ulysses_size": 1,
    "ring_size": 1,
    "n_prompt": "A close-up of a person talking." # Negative prompt
}

def main():
    """
    Connects to Redis and adds a single job to the audio_windows stream.
    """
    print("Connecting to Redis...")
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        print("Connected to Redis.")
    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        return

    job_id = str(uuid.uuid4())
    
    # This is the JSON structure the worker expects for the 'input.json' file
    input_data = {
        "prompt": "A person is talking, providing a clear and informative explanation.",
        "cond_video": COND_VIDEO_PATH,
        "cond_audio": {
            "person1": COND_AUDIO_PATH
        },
        "audio_type": "para", # 'para' for parallel audio, 'add' for sequential
        "video_audio": COND_AUDIO_PATH # Audio to be muxed into the final video
    }

    # This is the message payload for the Redis stream
    job_payload = {
        "job_id": job_id,
        "input_data": input_data,
        "args": GENERATION_ARGS
    }

    # The worker expects the 'data' field to be a JSON string
    message = {"data": json.dumps(job_payload)}

    try:
        message_id = r.xadd(AUDIO_STREAM, message)
        print(f"Successfully added job {job_id} to stream '{AUDIO_STREAM}' with message ID {message_id}")
    except Exception as e:
        print(f"Error adding job to stream: {e}")

if __name__ == "__main__":
    main()
