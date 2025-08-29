#!/usr/bin/env python3
"""
Windows용 Resume AI Filter API 서버 실행 스크립트
"""

import uvicorn
import multiprocessing
import sys

def get_worker_count():
    """CPU 코어 수에 따른 워커 수 계산"""
    cpu_count = multiprocessing.cpu_count()
    return max(1, min(cpu_count, 4))

def run_server(host="0.0.0.0", port=8000, workers=None, reload=False):
    """Windows에서 서버 실행"""
    
    if workers is None:
        workers = get_worker_count()
    
    print(f"Starting Resume AI Filter API server on Windows...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print(f"Reload: {reload}")
    print("-" * 50)
    
    try:
        # Windows에서는 uvicorn 직접 사용
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,  # reload 모드에서는 단일 워커
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume AI Filter API Server (Windows)")
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