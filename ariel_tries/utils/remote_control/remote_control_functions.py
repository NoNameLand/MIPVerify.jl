# Imports
import time
import subprocess

def run_batch_verification(command, success_string):
    timeout = 3600 # 1 hour
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout) # seconds
    except Exception:
        # timeout has reached
        return False, timeout / 60
    runtime = (time.time() - start_time) / 60
    status = True if success_string in result.stdout else False
    return status, runtime



status, runtime = run_batch_verification("julia ../main.jl", "INFEASIBLE")
print(status)
print(f"Time it took is: {runtime} seconds")