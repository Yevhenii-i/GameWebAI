import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from app.models.neural_net import GameAI
from app.services.vectorizer import vectorize_state
from app.services.action_map import get_action_index
from dotenv import load_dotenv
import os

load_dotenv()
ai_version_train = os.getenv('AI_VERSION_TRAINING')

def train_from_db(raw_matches):
    model = GameAI()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    states = []
    targets = []

    weights = torch.tensor([
        0.3,  # end_turn
        1.0,  # get_gold_2
        1.5,  # get_gold_4
        1.0,  # get_card_1
        1.5,  # get_card_2
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0  # play_cards ()
    ], dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights)

    for game in raw_matches:
        if "winner" not in game or "history" not in game:
            continue

        winner_id = game["winner"]

        for turn in game["history"]:

            # if winner_id == turn["active_actor"] and game["player_score"] != game["opponent_score"]:
                try:

                    v_state = vectorize_state(turn["state_before"])
                    action_idx = get_action_index(turn["action_taken"])

                    states.append(v_state)
                    targets.append(action_idx)
                except Exception as e:
                    continue

    if not states:
        print("No valid winning turns found to train on.")
        return

    X = torch.tensor(states, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.long)

    model.train()
    for epoch in range(101):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "app/services/ml_models/" + ai_version_train + ".pth")
    print("Model trained and saved to app/services/ml_models/" + ai_version_train + ".pth")