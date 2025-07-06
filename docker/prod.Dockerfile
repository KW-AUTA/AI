# 빌드 단계
FROM python:3.12.2-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install/packages -r requirements.txt

# 실제 실행 단계
FROM python:3.12.2-slim

WORKDIR /app

COPY --from=builder /install/packages /usr/local

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
