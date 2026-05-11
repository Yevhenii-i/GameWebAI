from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import ai, games
from app.db.database import create_db_and_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(title="Card Game AI Backend", lifespan=lifespan)

app.include_router(ai.router, prefix="/ai", tags=["AI Integration"])
app.include_router(games.router, prefix="/games", tags=["Data Collection"])

@app.get("/")
def read_root():
    return {"status": "Server is running", "game": "Turn-Based Card AI"}