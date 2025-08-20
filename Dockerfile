# Python 3.9 slim 이미지를 베이스로 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 패키지 설치 (개발 모드)
RUN pip install -e .

# 포트 설정 (필요시)
EXPOSE 8000

# 환경변수 설정
ENV PYTHONPATH=/app

# 기본 명령어 (나중에 API 서버나 CLI 도구로 변경 가능)
CMD ["python", "-c", "print('Generated Resume Filter is ready!')"]