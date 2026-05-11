import torch
import numpy as np

from fastapi import APIRouter
from app.schemas.game_schemas import StateSnapshot
from app.services.vectorizer import vectorize_state
from app.services.action_map import ACTION_SPACE, get_action_string
from app.models.neural_net import GameAI
from dotenv import load_dotenv
import os

router = APIRouter()
load_dotenv()
ai_version_in_use = os.getenv('AI_VERSION_IN_USE')
model = GameAI()

try:
    model.load_state_dict(torch.load("app/services/ml_models/" + ai_version_in_use + ".pth"))
    model.eval()
except:
    print("No trained model found.")

@router.post("/move")
async def get_ai_move(state: StateSnapshot):
    state_dict = state.model_dump()
    state_vector = vectorize_state(state_dict)

    with torch.no_grad():
        input_tensor = torch.tensor([state_vector], dtype=torch.float32)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    available_actions_from_godot = state.available_actions

    for i, action_name in enumerate(ACTION_SPACE):
        if action_name not in available_actions_from_godot:
            probs[i] = -1.0  # Mask it out

    best_action_index = int(np.argmax(probs))
    best_action_string = get_action_string(best_action_index)

    return {"status": "success", "chosen_action": best_action_string}

@router.post("/reinforce")
async def get_ai_move(state: StateSnapshot):
    state_dict = state.model_dump()
    state_vector = vectorize_state(state_dict)

    with torch.no_grad():
        input_tensor = torch.tensor([state_vector], dtype=torch.float32)
        logits = model(input_tensor)
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1).numpy()[0]

    available_actions_from_godot = state.available_actions

    for i, action_name in enumerate(ACTION_SPACE):
        #if action_name == 'end_turn':
        #    probs[i] = probs[i] / 10
        if action_name not in available_actions_from_godot:
            probs[i] = -1.0  # Mask it out


    valid_probs = probs[probs > 0]  # Filter out masked -1.0 values
    if np.sum(valid_probs) > 0:
        probs = np.where(probs > 0, probs / np.sum(probs[probs > 0]), 0)
    else:
        probs[0] = 1.0

    best_action_index = np.random.choice(len(ACTION_SPACE), p=probs)
    best_action_string = get_action_string(best_action_index)

    return {"status": "success", "chosen_action": best_action_string}