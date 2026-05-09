from sqlmodel import SQLModel, create_engine, Session, select
from app.models.game_record import GameRecord

# For local development, this creates a file called 'database.db' in your project root.
# For production, change this to your PostgreSQL URL: "postgresql://user:pass@localhost/dbname"
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args=connect_args)

def create_db_and_tables():
    # This reads your SQLModel classes and creates the tables in the database
    SQLModel.metadata.create_all(engine)

def get_session():
    # This is a dependency we will inject into our API endpoints
    with Session(engine) as session:
        yield session

def get_training_games_raw():
    with Session(engine) as session:
        statement = select(GameRecord).where(GameRecord.game_type == "training")
        results = session.exec(statement).all()
        # SQLModel automatically converts the JSON column back into a Python list
        return [game.model_dump() for game in results]

def get_reinforcement_games_raw():
    with Session(engine) as session:
        statement = select(GameRecord).where(GameRecord.game_type == "reinforcement")
        results = session.exec(statement).all()
        # SQLModel automatically converts the JSON column back into a Python list
        return [game.model_dump() for game in results]