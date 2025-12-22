import os
import redis
import time
import json
import sys
import subprocess
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Redis connection details
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
AUDIO_STREAM = "audio_windows"
VIDEO_STREAM = "mp4_ready"
CONSUMER_GROUP = "video_generators"
WORKER_ID = f"video_worker_{os.getpid()}"

# Paths for generate_infinitetalk.py
CKPT_DIR = os.getenv("CKPT_DIR")
INFINITETALK_DIR = os.getenv("INFINITETALK_DIR")
WAV2VEC_DIR = os.getenv("WAV2VEC_DIR")
AUDIO_SAVE_DIR = os.getenv("AUDIO_SAVE_DIR")

FAKE_RUN = os.getenv("FAKE_RUN", "true").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
ALLOW_GENERATION = os.getenv("ALLOW_GENERATION", "false").lower() == "true"

def cleanup_output_dir(output_dir, max_files=5):
    """
    Keep only the most recent max_files in the output directory.
    """
    try:
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
        if len(files) <= max_files:
            return
        
        # Sort files by modification time (oldest first)
        files.sort(key=os.path.getmtime)
        
        # Remove oldest files
        for f in files[:-max_files]:
            try:
                os.remove(f)
                print(f"Cleaned up old result: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")
    except Exception as e:
        print(f"Error during output directory cleanup: {e}")

def main():
    """
    Connects to Redis, creates a consumer group, and listens for new messages
    on the audio_windows stream.
    """
    print("Starting video worker...")
    # Retry loop for initial Redis connection
    max_retries = 10
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            print("Connected to Redis.")
            break
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"Error connecting to Redis (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10)  # Exponential backoff cap at 10s
            else:
                print(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                return

    # Create the consumer group if it doesn't exist
    try:
        r.xgroup_create(AUDIO_STREAM, CONSUMER_GROUP, id='0', mkstream=True)
        print(f"Consumer group '{CONSUMER_GROUP}' created for stream '{AUDIO_STREAM}'.")
    except redis.exceptions.ResponseError as e:
        if "name already exists" in str(e):
            print(f"Consumer group '{CONSUMER_GROUP}' already exists.")
        else:
            print(f"Error creating consumer group: {e}")
            return

    print(f"Worker '{WORKER_ID}' is waiting for messages...")

    while True:
        try:
            messages = r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=WORKER_ID,
                streams={AUDIO_STREAM: '>'},
                count=1,
                block=5000
            )

            if not messages:
                continue

            for stream, message_list in messages:
                for message_id, data in message_list:
                    temp_dir = None
                    job_id = "unknown"
                    try:
                        print(f"Received message {message_id}")
                        job_data = json.loads(data['data'])
                        job_id = job_data['job_id']
                        input_data = job_data['input_data']
                        args = job_data['args']

                        temp_dir = tempfile.mkdtemp()
                        input_json_path = os.path.join(temp_dir, 'input.json')
                        with open(input_json_path, 'w') as f:
                            json.dump(input_data, f)

                        output_filename = f"{job_id}.mp4"
                        output_path = os.path.join(temp_dir, output_filename)

                        is_dryrun = args.get("task") == "dryrun"

                        if not is_dryrun and not ALLOW_GENERATION:
                            print("🚫 Generation disabled — refusing non-dryrun task")
                            # relying on finally block for xack if we continue, or explicit here is fine too
                            r.xack(AUDIO_STREAM, CONSUMER_GROUP, message_id)
                            continue

                        command = [
                            sys.executable,
                            'InfiniteTalk/generate_infinitetalk.py',
                            '--input_json', input_json_path,
                            '--ckpt_dir', CKPT_DIR,
                            '--infinitetalk_dir', INFINITETALK_DIR,
                            '--wav2vec_dir', WAV2VEC_DIR,
                            '--audio_save_dir', AUDIO_SAVE_DIR,
                            '--save_file', output_path,
                            '--size', str(args['size']),
                            '--motion_frame', str(args['motion_frame']),
                            '--frame_num', str(args['frame_num']),
                            '--sample_shift', str(args['sample_shift']),
                            '--sample_steps', str(args['sd_steps']),
                            '--sample_text_guide_scale', str(args['text_guide_scale']),
                            '--sample_audio_guide_scale', str(args['audio_guide_scale']),
                            '--base_seed', str(args['seed']),
                            '--color_correction_strength', str(args['color_correction_strength']),
                            '--task', str(args['task']),
                            '--ulysses_size', str(args['ulysses_size']),
                            '--ring_size', str(args['ring_size']),
                        ]
                        
                        if args.get('n_prompt') is not None:
                            command.extend(['--n_prompt', str(args['n_prompt'])])

                        print(f"Running command: {' '.join(command)}")
                        
                        if FAKE_RUN or is_dryrun:
                            print(f"🧪 FAKE RUN for job {job_id}")

                            src = "/app/src/hls/idle_loop.mp4"
                            output_dir = "output"
                            os.makedirs(output_dir, exist_ok=True)

                            final_video_path = os.path.join(output_dir, f"{job_id}.mp4")
                            
                            # Try to use the real idle loop if available, else fallback to dummy
                            if os.path.exists(src):
                                shutil.copy(src, final_video_path)
                            else:
                                print(f"⚠️ Warning: Mock video source {src} not found. Using dummy data.")
                                with open(final_video_path, "wb") as f:
                                    f.write(b"FAKE_MP4_DATA")

                            r.xadd(VIDEO_STREAM, {
                                "job_id": job_id,
                                "video_path": final_video_path
                            })

                            r.set(f"result:{job_id}", final_video_path, ex=3600)
                            r.xack(AUDIO_STREAM, CONSUMER_GROUP, message_id)
                            print(f"FAKE video published for {job_id}")
                            
                            cleanup_output_dir(output_dir)
                            continue

                        # Real execution
                        subprocess.run(command, check=True, capture_output=True, text=True)

                        # The output path from the script includes the extension, but save_file does not.
                        video_path = output_path
                        
                        if os.path.exists(video_path):
                            output_dir = 'output'
                            os.makedirs(output_dir, exist_ok=True)
                            final_video_path = os.path.join(output_dir, output_filename)
                            shutil.copy(video_path, final_video_path)
                            
                            # Publish to stream for general notifications
                            r.xadd(VIDEO_STREAM, {'job_id': job_id, 'video_path': final_video_path})
                            print(f"Published result for job {job_id} to stream '{VIDEO_STREAM}'")

                            # Set result in a job-specific key for polling
                            result_key = f"result:{job_id}"
                            r.set(result_key, final_video_path, ex=3600) # 1-hour expiration
                            print(f"Set result for job {job_id} in key '{result_key}'")
                            
                            # Clean up old results
                            cleanup_output_dir(output_dir)

                        else:
                            print(f"Error: Video file not found at {video_path}")

                    except subprocess.CalledProcessError as e:
                        print(f"Error generating video for job {job_id}: {e}")
                        print(f"Stdout: {e.stdout}")
                        print(f"Stderr: {e.stderr}")

                    except Exception as e:
                        print(f"Job {job_id} failed: {e}")

                    finally:
                        if temp_dir and os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                        
                        r.xack(AUDIO_STREAM, CONSUMER_GROUP, message_id)
                        print(f"Acknowledged message {message_id}.")

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()