import numpy as np
from typing import Dict, Any

NUM_UNIQUE_CARDS = 11

def vectorize_state(snapshot: Dict[str, Any]) -> np.ndarray:
    scalars = [
        float(snapshot["round"]),
        float(snapshot["active_character"]),
        float(snapshot["self_gold"]),
        float(snapshot["self_score"]),
        float(snapshot["self_hand_size"]),
        float(snapshot["self_board_size"]),
        float(snapshot["opponent_gold"]),
        float(snapshot["opponent_score"]),
        float(snapshot["opponent_hand_size"]),
        float(snapshot["opponent_board_size"])
    ]

    self_board_counts = [0.0] * NUM_UNIQUE_CARDS
    for card_id in snapshot["self_active_cards"]:
        self_board_counts[card_id] += 1.0

    opponent_board_counts = [0.0] * NUM_UNIQUE_CARDS
    for card_id in snapshot["opponent_active_cards"]:
        opponent_board_counts[card_id] += 1.0

    hand_counts = [float(count) for count in snapshot["hand_card_records"]]

    card_features = [0.0] * (NUM_UNIQUE_CARDS * 3)
    for record in snapshot["unique_hand_card_records"]:
        c_id = record["card_id"]
        base_idx = c_id * 3
        card_features[base_idx] = float(record["current_cost"])
        card_features[base_idx + 1] = float(record["current_value"])
        card_features[base_idx + 2] = float(record["potential_value"])

    final_vector = scalars + self_board_counts + opponent_board_counts + hand_counts + card_features
    return np.array(final_vector, dtype=np.float32)