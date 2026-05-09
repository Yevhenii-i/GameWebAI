from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from app.db.database import get_session
from app.models.game_record import GameRecord
from app.schemas.game_schemas import GameSavePayload

router = APIRouter()


@router.post("/save")
async def save_game(payload: GameSavePayload, session: Session = Depends(get_session)):
    try:
        # Convert the Pydantic payload from Godot into an SQLModel database row
        new_game = GameRecord(
            game_type = payload.game_type,
            player_score=payload.player_score,
            opponent_score=payload.opponent_score,
            winner=payload.winner,
            total_rounds=payload.total_rounds,
            history=payload.history
        )

        # Add to session and commit to the database
        session.add(new_game)
        session.commit()
        session.refresh(new_game)  # Refreshes to get the newly generated ID

        return {"status": "success", "game_id": new_game.id}

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save game: {str(e)}")