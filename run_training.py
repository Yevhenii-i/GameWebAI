import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import get_training_games_raw
from app.services.trainer import train_from_db


def main():
    print("Loading games from database.db...")

    try:
        raw_matches = get_training_games_raw()
    except Exception as e:
        print(f"Database error: {e}")
        return

    count = len(raw_matches)
    if count == 0:
        print("Database is empty. Upload some games from Godot first!")
        return

    print(f"Found {count} matches. Training in progress...")

    # This calls the trainer function we wrote earlier
    train_from_db(raw_matches)


if __name__ == "__main__":
    main()