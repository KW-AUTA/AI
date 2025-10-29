# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# NumPy와 PyTorch 호환성을 위한 환경 변수 설정
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from fastapi import FastAPI
from routes import test_route, evaluate
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from service.resource_manager import resource_manager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 생명주기 관리 - 시작 및 종료 시 리소스 관리"""
    # 서버 시작 시
    logger.info("Starting server and initializing resources...")
    try:
        resource_manager.initialize()
        logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

    yield

    # 서버 종료 시
    logger.info("Shutting down server...")
    resource_manager.shutdown()
    logger.info("Server shutdown complete")


app = FastAPI(lifespan=lifespan)
app.include_router(test_route.router)
app.include_router(evaluate.router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return {"message": "Welcome to the API"}


@app.get("/health")
async def health():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "ray_initialized": resource_manager.ray_initialized,
        "navigator_ready": resource_manager.navigator is not None
    }