from fastapi import FastAPI
from routes import test_route
from config import get_settings
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
settings = get_settings()

# YOLO 라우터 추가
app.include_router(test_route.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

# 전역 예외 핸들러 추가
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler caught: {str(exc)}")
    logger.error(f"Error type: {type(exc)}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    return {"error": str(exc)}