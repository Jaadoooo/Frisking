# Run this script and wait for the output_video file to generate 
# Make sure to set the correct path of the input video in script 1 and 4.

import subprocess
import sys

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    scripts = [
        "1Detectron.py",  # replace with the actual script names
        "2YOLO.py",
        "3Filter.py",
        "4Final.py"
    ]

    for script in scripts:
        run_script(script)