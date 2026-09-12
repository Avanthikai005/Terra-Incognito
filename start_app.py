#!/usr/bin/env python
"""Terra Incognita — Unified Launcher

Starts both:
1. PyTorch Model API Server (FastAPI / Uvicorn on http://127.0.0.1:8000)
2. React Frontend Dev Server (Vite on http://127.0.0.1:5173)
"""
import os
import signal
import subprocess
import sys
import time

PYTHON_EXE = r"C:\Users\NIVA KALYANI H\AppData\Local\Programs\Python\Python313\python.exe"
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")


def main():
    print("=" * 65)
    print("  Terra Incognita — Continual Learning & Disaster Response")
    print("=" * 65)
    print(f">> Root:     {ROOT_DIR}")
    print(f">> Python:   {PYTHON_EXE}")

    procs = []

    try:
        # 1. Start Python API Backend Server
        print("\n[1/2] Starting PyTorch Model API Server on http://127.0.0.1:8000 ...")
        api_env = os.environ.copy()
        api_env["PYTHONPATH"] = ROOT_DIR
        api_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "uvicorn", "src.api_server:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT_DIR,
            env=api_env,
        )
        procs.append(api_proc)

        # Wait a moment for API to initialize
        time.sleep(2)

        # 2. Start Vite Frontend Server
        print("\n[2/2] Starting React + Vite Frontend UI on http://127.0.0.1:5173 ...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
            cwd=FRONTEND_DIR,
        )
        procs.append(frontend_proc)

        print("\n" + "=" * 65)
        print("  APPLICATION READY!")
        print("  - Frontend UI:  http://127.0.0.1:5173")
        print("  - Model API:    http://127.0.0.1:8000/docs")
        print("  Press Ctrl+C to terminate all services.")
        print("=" * 65 + "\n")

        # Keep running until user terminates
        while True:
            time.sleep(1)
            for p in procs:
                if p.poll() is not None:
                    print(f"\n[!] Process {p.pid} exited with code {p.returncode}")
                    return

    except KeyboardInterrupt:
        print("\n>> Shutting down services...")
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
