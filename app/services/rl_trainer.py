import torch
import torch.optim as optim
from torch.distributions import Categorical
from app.models.neural_net import GameAI
from app.services.vectorizer import vectorize_state
from app.services.action_map import get_action_index


def train_self_play_rl(raw_matches):
    model = GameAI()

    # Load your Behavioral Cloning weights as the starting point!
    try:
        model.load_state_dict(torch.load("app/services/ml_models/experimental_v5.pth"))
        print("Loaded BC model. Transitioning to RL...")
    except:
        print("Starting RL from scratch (random weights).")

    # Use a much smaller learning rate for RL to prevent catastrophic forgetting
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
            # Who took this action? Did they ultimately win or lose?
            actor_id = turn["active_actor"]

            # Sparse Reward: +1 if this actor won the game, -1 if they lost
            reward = 1.0 if actor_id == winner_id else -1.0

            try:
                v_state = vectorize_state(turn["state_before"])
                action_idx = get_action_index(turn["action_taken"])
            except Exception:
                continue

            # 1. Forward Pass
            state_tensor = torch.tensor([v_state], dtype=torch.float32)
            logits = model(state_tensor)

            # 2. Create a Probability Distribution
            dist = Categorical(logits=logits)

            # 3. Calculate the Log Probability of the action that was actually taken
            action_tensor = torch.tensor([action_idx], dtype=torch.long)
            log_prob = dist.log_prob(action_tensor)

            # 4. Policy Gradient Math: Multiply by reward and negate for gradient ascent
            loss = -(log_prob * reward)
            loss.backward()  # Accumulate gradients

            total_loss += loss.item()
            batch_count += 1

            # Update weights every 256 actions to keep memory usage low
            if batch_count % 256 == 0:
                optimizer.step()
                optimizer.zero_grad()

    # Final step for any remaining gradients
    if batch_count % 256 != 0:
        optimizer.step()

    print(f"RL Training pass complete. Average Loss metric: {total_loss / max(1, batch_count):.4f}")
    torch.save(model.state_dict(), "app/services/ml_models/experimental_v5_1.pth")