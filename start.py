import subprocess
import sys
import os

def main():
    # Read dynamic PORT from environment (default to 8000)
    port = os.environ.get("PORT", "8000")
    
    print("Starting Celery worker process in background...", flush=True)
    worker_cmd = [
        "celery", "-A", "medscan.tasks.celery_app.celery_app", 
        "worker", "--loglevel=info", "-Q", "pipeline", "--pool=solo"
    ]
    # Launch celery worker
    worker_proc = subprocess.Popen(worker_cmd)
    
    print(f"Starting FastAPI web application on port {port}...", flush=True)
    api_cmd = [
        "uvicorn", "medscan.api.main:app", 
        "--host", "0.0.0.0", "--port", port
    ]
    
    try:
        # Launch FastAPI and keep it in the foreground
        subprocess.run(api_cmd, check=True)
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...", flush=True)
    except Exception as e:
        print(f"FastAPI application exited with error: {e}", file=sys.stderr, flush=True)
    finally:
        print("Terminating Celery worker process...", flush=True)
        worker_proc.terminate()
        try:
            worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force-killing Celery worker...", flush=True)
            worker_proc.kill()
            worker_proc.wait()
        print("Services stopped.", flush=True)

if __name__ == "__main__":
    main()
