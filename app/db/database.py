from sqlmodel import SQLModel, create_engine, Session, select
from app.models.game_record import GameRecord
from dotenv import load_dotenv
import os

load_dotenv()
database_url = os.getenv('DATABASE_URL')
# For local development, this creates a file called 'database.db' in your project root.
# For production, change this to your PostgreSQL URL: "postgresql://user:pass@localhost/dbname"
#sqlite_file_name = "database.db"
sqlite_url = "database_url"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args=connect_args)

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