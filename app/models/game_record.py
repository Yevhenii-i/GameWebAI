from sqlmodel import SQLModel, Field, Column, JSON
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


class GameRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    game_type: Optional[str] = Field(default="training", nullable=True)
    player_score: int
    opponent_score: int
    winner: int  # 0 for first participant, 1 for second participant, 2 for draw
    total_rounds: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    history: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))