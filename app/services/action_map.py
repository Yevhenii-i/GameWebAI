NUM_UNIQUE_CARDS = 11

ACTION_SPACE = [
    "end_turn",
    "get_gold_2",
    "get_gold_4",
    "get_card_1",
    "get_card_2"
] + [f"play_card_{i}" for i in range(NUM_UNIQUE_CARDS)]

def get_action_string(index: int) -> str:
    return ACTION_SPACE[index]

def get_action_index(action_str: str) -> int:
    return ACTION_SPACE.index(action_str)