"""From-scratch DQN, Double DQN, and Double DQN with PER for LunarLander-v2.

The script writes episode-level returns, fixed-state Q estimates, and mean
bootstrap targets to a CSV. It intentionally uses no RL implementation package.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ENV_ID = "LunarLander-v2"
BUFFER_CAPACITY = 50_000
BATCH_SIZE = 64
TARGET_UPDATE_EVERY = 500
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10_000
GAMMA = 0.99
LEARNING_RATE = 1e-3
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_PRIORITY_EPS = 1e-6
ALGORITHMS = ("dqn", "double_dqn", "ddqn_per")


class QNetwork(nn.Module):
    """MLP with the requested two 128-unit ReLU hidden layers."""

    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    indexes: np.ndarray
    weights: np.ndarray


class SumTree:
    """Array-backed sum tree for O(log capacity) proportional PER sampling."""

    def __init__(self, capacity: int) -> None:
        leaf_count = 1
        while leaf_count < capacity:
            leaf_count *= 2
        self.leaf_count = leaf_count
        self.values = np.zeros(2 * leaf_count, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.values[1])

    def update(self, index: int, value: float) -> None:
        node = self.leaf_count + index
        delta = value - self.values[node]
        while node:
            self.values[node] += delta
            node //= 2

    def find_prefix_sum(self, mass: float) -> int:
        node = 1
        while node < self.leaf_count:
            left = 2 * node
            if mass <= self.values[left]:
                node = left
            else:
                mass -= self.values[left]
                node = left + 1
        return node - self.leaf_count


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, prioritized: bool, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.prioritized = prioritized
        self.rng = rng
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.tree = SumTree(capacity) if prioritized else None
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool) -> None:
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.terminated[index] = float(terminated)
        if self.tree is not None:
            self.tree.update(index, self.max_priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 1.0) -> Batch:
        if self.size < batch_size:
            raise ValueError("Not enough replay transitions to sample a minibatch")
        if self.tree is None:
            indexes = self.rng.choice(self.size, size=batch_size, replace=False)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            total = self.tree.total
            if total <= 0:
                raise RuntimeError("PER sum tree has no positive priorities")
            segment = total / batch_size
            masses = (np.arange(batch_size) + self.rng.random(batch_size)) * segment
            indexes = np.asarray([self.tree.find_prefix_sum(float(x)) for x in masses], dtype=np.int64)
            indexes = np.minimum(indexes, self.size - 1)
            probabilities = np.asarray(
                [self.tree.values[self.tree.leaf_count + int(i)] / total for i in indexes], dtype=np.float64
            )
            weights = (self.size * probabilities) ** (-beta)
            weights /= weights.max()
            weights = weights.astype(np.float32)

        return Batch(
            states=self.states[indexes],
            actions=self.actions[indexes],
            rewards=self.rewards[indexes],
            next_states=self.next_states[indexes],
            terminated=self.terminated[indexes],
            indexes=indexes,
            weights=weights,
        )

    def update_priorities(self, indexes: np.ndarray, td_errors: np.ndarray) -> None:
        if self.tree is None:
            return
        for index, error in zip(indexes, td_errors, strict=True):
            priority = (abs(float(error)) + PER_PRIORITY_EPS) ** PER_ALPHA
            self.tree.update(int(index), priority)
            self.max_priority = max(self.max_priority, priority)


def epsilon_at(step: int) -> float:
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def make_fixed_eval_states(state_dim: int, count: int = 100) -> np.ndarray:
    """Use the same 100 seeded initial states for every method and seed."""
    env = gym.make(ENV_ID)
    states = []
    try:
        for i in range(count):
            state, _ = env.reset(seed=910_000 + i)
            states.append(state)
    finally:
        env.close()
    result = np.asarray(states, dtype=np.float32)
    if result.shape != (count, state_dim):
        raise RuntimeError(f"Unexpected evaluation state shape: {result.shape}")
    return result


def train_one(algorithm: str, seed: int, max_steps: int, device: torch.device, eval_states: np.ndarray) -> list[dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = gym.make(ENV_ID)
    try:
        state_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)
        if state_dim != 8 or n_actions != 4:
            raise ValueError(f"Expected LunarLander state/action dimensions 8/4, got {state_dim}/{n_actions}")
        env.action_space.seed(seed)
        rng = np.random.default_rng(seed)
        replay = ReplayBuffer(BUFFER_CAPACITY, state_dim, algorithm == "ddqn_per", rng)
        online = QNetwork(state_dim, n_actions).to(device)
        target = QNetwork(state_dim, n_actions).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()
        optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
        fixed_states = torch.as_tensor(eval_states, dtype=torch.float32, device=device)

        rows: list[dict] = []
        env_steps = 0
        episode = 0
        state, _ = env.reset(seed=seed)
        episode_return = 0.0
        episode_target_sum = 0.0
        episode_updates = 0

        while env_steps < max_steps:
            epsilon = epsilon_at(env_steps)
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(online(state_tensor).argmax(dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            replay.add(state, action, float(reward), next_state, terminated)
            env_steps += 1

            if replay.size >= BATCH_SIZE:
                beta = PER_BETA_START + min(env_steps / max_steps, 1.0) * (1.0 - PER_BETA_START)
                batch = replay.sample(BATCH_SIZE, beta)
                states = torch.as_tensor(batch.states, dtype=torch.float32, device=device)
                actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
                next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=device)
                terminated_t = torch.as_tensor(batch.terminated, dtype=torch.float32, device=device)
                weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=device)

                chosen_q = online(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    if algorithm == "dqn":
                        next_q = target(next_states).max(dim=1).values
                    else:
                        best_actions = online(next_states).argmax(dim=1, keepdim=True)
                        next_q = target(next_states).gather(1, best_actions).squeeze(1)
                    targets = rewards + GAMMA * (1.0 - terminated_t) * next_q
                td_errors = targets - chosen_q
                per_sample_loss = F.smooth_l1_loss(chosen_q, targets, reduction="none")
                loss = (weights * per_sample_loss).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
                optimizer.step()
                if algorithm == "ddqn_per":
                    replay.update_priorities(batch.indexes, td_errors.detach().abs().cpu().numpy())
                episode_target_sum += float(targets.mean().item())
                episode_updates += 1

            if env_steps % TARGET_UPDATE_EVERY == 0:
                target.load_state_dict(online.state_dict())

            state = next_state
            if terminated or truncated:
                with torch.no_grad():
                    q_eval_mean = float(online(fixed_states).max(dim=1).values.mean().item())
                episode += 1
                rows.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "episode": episode,
                        "env_steps": env_steps,
                        "episode_return": episode_return,
                        "q_eval_mean": q_eval_mean,
                        "target_mean": episode_target_sum / episode_updates if episode_updates else float("nan"),
                        "epsilon_end": epsilon_at(env_steps),
                        "completed": True,
                    }
                )
                state, _ = env.reset()
                episode_return = 0.0
                episode_target_sum = 0.0
                episode_updates = 0

        return rows
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=100_000, help="Environment transitions per algorithm and seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 42])
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--output", type=Path, default=Path("results/task2_metrics.csv"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    eval_states = make_fixed_eval_states(8, count=100)

    all_rows: list[dict] = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            print(f"Starting algorithm={algorithm} seed={seed} max_steps={args.max_steps}", flush=True)
            rows = train_one(algorithm, seed, args.max_steps, device, eval_states)
            all_rows.extend(rows)
            completed = [row["episode_return"] for row in rows]
            tail = float(np.mean(completed[-20:])) if completed else float("nan")
            print(f"Finished algorithm={algorithm} seed={seed} episodes={len(rows)} last20_return={tail:.1f}", flush=True)

    fields = [
        "algorithm", "seed", "episode", "env_steps", "episode_return", "q_eval_mean",
        "target_mean", "epsilon_end", "completed",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} complete-episode records to {args.output}", flush=True)


if __name__ == "__main__":
    main()
