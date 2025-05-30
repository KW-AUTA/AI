from fastapi import FastAPI
from routes import test_route
app = FastAPI()
app.include_router(test_route.router)

@app.get("/")
async def root():
    return "서버 가동 중입니다!"