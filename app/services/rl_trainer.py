import torch
import torch.optim as optim
from torch.distributions import Categorical
from app.models.neural_net import GameAI
from app.services.vectorizer import vectorize_state
from app.services.action_map import get_action_index
from dotenv import load_dotenv
import os

load_dotenv()
ai_version_in_use = os.getenv('AI_VERSION_IN_USE')
ai_version_train = os.getenv('AI_VERSION_TRAINING')

def train_self_play_rl(raw_matches):
    model = GameAI()

    try:
        model.load_state_dict(torch.load("app/ml_models/" + ai_version_in_use + ".pth"))
        print("Loaded BC model. Transitioning to RL...")
    except:
        print("Starting RL from scratch (random weights).")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()

    total_loss = 0
    batch_count = 0

    optimizer.zero_grad()

    for game in raw_matches:
        if "winner" not in game:
            continue

        winner_id = game["winner"]

        if winner_id == 2: #skip draw matches
            continue

        for turn in game["history"]:
            actor_id = turn["active_actor"]

            reward = 1.0 if actor_id == winner_id else -1.0

            try:
                v_state = vectorize_state(turn["state_before"])
                action_idx = get_action_index(turn["action_taken"])
            except Exception:
                continue

            state_tensor = torch.tensor([v_state], dtype=torch.float32)
            logits = model(state_tensor)

            dist = Categorical(logits=logits)

            action_tensor = torch.tensor([action_idx], dtype=torch.long)
            log_prob = dist.log_prob(action_tensor)

            loss = -(log_prob * reward)
            loss.backward()

            total_loss += loss.item()
            batch_count += 1

            if batch_count % 256 == 0:
                optimizer.step()
                optimizer.zero_grad()

    if batch_count % 256 != 0:
        optimizer.step()

    print(f"RL Training pass complete. Average Loss metric: {total_loss / max(1, batch_count):.4f}")
    torch.save(model.state_dict(), "app/ml_models/" + ai_version_train + ".pth")