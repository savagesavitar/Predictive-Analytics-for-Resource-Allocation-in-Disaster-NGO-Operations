"""
Launch script for the Odisha Flood Resource Planner Dashboard.
"""
import os
import sys
import subprocess

if __name__ == '__main__':
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')
    print("Starting Odisha Flood Resource Planner Dashboard...")
    print(f"Dashboard: http://localhost:8501")
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])
