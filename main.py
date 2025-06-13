from fastapi import FastAPI
from routes import test_route
import logging

app = FastAPI()
app.include_router(test_route.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}