# NumPy와 PyTorch 호환성을 위한 환경 변수 설정
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from fastapi import FastAPI
from routes import test_route, evaluate
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.include_router(test_route.router)
app.include_router(evaluate.router)

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
async def root():
    return {"message": "Welcome to the API"}