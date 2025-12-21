import subprocess
import time
import sys
import urllib.request
import urllib.error

def check_docker_services():
    print("Checking docker services status...")
    try:
        # Run docker-compose ps to see container status
        result = subprocess.run(["docker-compose", "ps", "--format", "json"], capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback for older docker-compose or different format
            result = subprocess.run(["docker-compose", "ps"], capture_output=True, text=True)
            print(result.stdout)
        else:
            print(result.stdout)
            
        print("\nVerifying containers are running...")
        # Simple check if "Exit" is in output, which usually indicates a stopped container in standard view
        # or checking specific states if we parsed JSON (but kept simple for now)
        if "Exit" in result.stdout:
            print("WARNING: Some containers seem to have exited.")
    except FileNotFoundError:
        print("Error: docker-compose not found.")
        return False
    return True

def poll_api_health(url="http://localhost:8000/health", retries=5, delay=2):
    print(f"Polling API health at {url}...")
    for i in range(retries):
        try:
            with urllib.request.urlopen(url) as response:
                if response.status == 200:
                    data = response.read()
                    print(f"Health check passed: {data.decode('utf-8')}")
                    return True
        except urllib.error.URLError as e:
            print(f"Attempt {i+1}/{retries} failed: {e}")
        except Exception as e:
            print(f"Attempt {i+1}/{retries} failed with error: {e}")
        
        time.sleep(delay)
    
    print("API health check failed after multiple retries.")
    return False

if __name__ == "__main__":
    if not check_docker_services():
        sys.exit(1)
        
    print("-" * 20)
    
    if poll_api_health():
        print("System appears healthy.")
        sys.exit(0)
    else:
        print("System health check failed.")
        sys.exit(1)
