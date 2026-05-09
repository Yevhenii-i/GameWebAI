from pydantic import BaseModel
from typing import List, Dict, Any

class GameSavePayload(BaseModel):
    game_type: str
    player_score: int
    opponent_score: int
    winner: int
    total_rounds: int
    history: List[Dict[str, Any]]

class CardRecord(BaseModel):
    card_id: int
    current_cost: int
    current_value: int
    potential_value: int

class StateSnapshot(BaseModel):
    round: int
    active_character: int
    self_gold: int
    self_score: int
    self_hand_size: int
    self_board_size: int
    opponent_gold: int
    opponent_score: int
    opponent_hand_size: int
    opponent_board_size: int
    self_active_cards: List[int]
    opponent_active_cards: List[int]
    hand_card_records: List[int]
    unique_hand_card_records: List[CardRecord]
    available_actions: List[str]
