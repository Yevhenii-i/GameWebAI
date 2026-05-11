from sqlmodel import SQLModel, create_engine, Session, select
from app.models.game_record import GameRecord
from dotenv import load_dotenv
import os

load_dotenv()
database_url = os.getenv('DATABASE_URL')

connect_args = {}
engine = create_engine(database_url, echo=False, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

def get_training_games_raw():
    with Session(engine) as session:
        statement = select(GameRecord).where(GameRecord.game_type == "training")
        results = session.exec(statement).all()
        return [game.model_dump() for game in results]

def get_reinforcement_games_raw():
    with Session(engine) as session:
        statement = select(GameRecord).where(GameRecord.game_type == "reinforcement")
        results = session.exec(statement).all()
        return [game.model_dump() for game in results]