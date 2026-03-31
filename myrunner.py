"""
Self-play training runner for 6x6 Checkers.
Trains a single Actor-Critic agent by having it play both sides of every episode.
"""

import mycheckersenv
import numpy as np
from myagent import ActorCriticAgent

NUM_EPISODES = 5000
SAVE_INTERVAL = 500
LOG_INTERVAL = 100
CHECKPOINT_PATH = "checkers_agent.pt"

mycheckersenv.CHECK_LEGALITY = False


def run_episode(env, agent):
    """
    Runs one self-play episode and updates the agent.

    Uses a pending dict to correctly align rewards with the actions that caused them.
    In AEC, env.last() returns the reward for the previous action, not the next one,
    so we hold each player's training tensors until their reward arrives on the next visit.
    """
    env.reset()
    agent.reset_trajectory()

    pending = {}
    episode_rewards = {"player_0": 0.0, "player_1": 0.0}

    for current_agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        episode_rewards[current_agent] += reward

        if current_agent in pending:
            log_prob, value, entropy = pending.pop(current_agent)
            agent._log_probs.append(log_prob)
            agent._values.append(value)
            agent._entropies.append(entropy)
            agent._rewards.append(reward)

        if termination or truncation:
            env.step(None)
            continue

        mask   = info["action_mask"]
        action = agent.select_action(obs, mask)

        pending[current_agent] = (
            agent._log_probs.pop(),
            agent._values.pop(),
            agent._entropies.pop(),
        )

        env.step(action)

    # flush any remaining pending transitions
    for a, (log_prob, value, entropy) in pending.items():
        agent._log_probs.append(log_prob)
        agent._values.append(value)
        agent._entropies.append(entropy)
        agent._rewards.append(0.0)

    r0 = episode_rewards["player_0"]
    r1 = episode_rewards["player_1"]
    if r0 > r1:
        winner = "player_0"
    elif r1 > r0:
        winner = "player_1"
    else:
        winner = "draw"

    return agent.finish_episode(), winner


def render_sample_game(agent):
    print("\n" + "=" * 50)
    print("SAMPLE GAME — trained agent (greedy) vs itself")
    print("=" * 50)

    env = mycheckersenv.env(render_mode="human")
    env.reset()
    total_rewards = {"player_0": 0.0, "player_1": 0.0}

    for current_agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        total_rewards[current_agent] += reward
        if termination or truncation:
            env.step(None)
            continue
        env.step(agent.select_greedy_action(obs, info["action_mask"]))

    print(f"Final cumulative rewards: {total_rewards}")
    env.close()


def train():
    env   = mycheckersenv.env()
    agent = ActorCriticAgent()

    win_counts   = {"player_0": 0, "player_1": 0, "draw": 0}
    metric_accum = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "mean_return": 0.0}

    print(f"Starting self-play training for {NUM_EPISODES} episodes...\n")

    for episode in range(1, NUM_EPISODES + 1):
        metrics, winner = run_episode(env, agent)
        win_counts[winner] += 1

        for k in metric_accum:
            metric_accum[k] += metrics.get(k, 0.0)

        if episode % LOG_INTERVAL == 0:
            n = LOG_INTERVAL
            print(
                f"Episode {episode:5d} | "
                f"p0 wins: {win_counts['player_0']:4d}  "
                f"p1 wins: {win_counts['player_1']:4d}  "
                f"draws: {win_counts['draw']:4d} | "
                f"actor_loss: {metric_accum['actor_loss']/n:.4f}  "
                f"critic_loss: {metric_accum['critic_loss']/n:.4f}  "
                f"entropy: {metric_accum['entropy']/n:.4f}  "
                f"mean_return: {metric_accum['mean_return']/n:.4f}"
            )
            metric_accum = {k: 0.0 for k in metric_accum}

        if episode % SAVE_INTERVAL == 0:
            agent.save(CHECKPOINT_PATH)

    agent.save(CHECKPOINT_PATH)
    print(f"\nTraining complete.")
    print(f"Total outcomes — {win_counts}")

    render_sample_game(agent)


if __name__ == "__main__":
    train()