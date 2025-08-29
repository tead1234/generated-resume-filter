#!/usr/bin/env python3
"""
Resume AI Filter API 서버 실행 스크립트
"""

import os
import sys
import subprocess
import multiprocessing

def get_worker_count():
    """CPU 코어 수에 따른 워커 수 계산"""
    cpu_count = multiprocessing.cpu_count()
    # CPU 코어 수의 2배 또는 최소 2개
    return max(2, min(cpu_count * 2, 8))

def run_server(host="0.0.0.0", port=8000, workers=None, reload=False):
    """서버 실행"""
    
    if workers is None:
        workers = get_worker_count()
    
    print(f"Starting Resume AI Filter API server...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print(f"Reload: {reload}")
    print("-" * 50)
    
    # gunicorn 명령어 구성
    cmd = [
        "gunicorn",
        "src.api.main:app",
        "--bind", f"{host}:{port}",
        "--workers", str(workers),
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--access-logfile", "-",
        "--error-logfile", "-",
        "--log-level", "info",
    ]
    
    # 개발 모드시 reload 옵션 추가
    if reload:
        cmd.extend(["--reload"])
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except FileNotFoundError:
        print("Error: gunicorn not found. Please install requirements:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume AI Filter API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )