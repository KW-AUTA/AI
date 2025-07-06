FROM python:3.12.2-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=0

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 캐시 활용을 위해 requirements만 먼저 복사
COPY requirements.txt .

# pip cache 디렉토리 환경 변수 설정
ENV PIP_CACHE_DIR=/root/.cache/pip

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 전체 소스 복사
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
