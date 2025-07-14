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