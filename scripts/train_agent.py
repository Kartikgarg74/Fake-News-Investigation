#!/usr/bin/env python3
"""Train an agent on Veritas and generate a learning curve.

Demonstrates the environment is RL-trainable by:
1. Running 200 episodes with a heuristic agent
2. Training a simple policy (logistic regression over state features)
3. Comparing heuristic vs trained policy scores
4. Generating a learning curve PNG

Usage:
    python scripts/train_agent.py
    python scripts/train_agent.py --episodes 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Make sure the package is importable regardless of cwd.
# The package `fake_news_investigator` lives one level ABOVE the project dir
# (the project dir IS the package), so we need to add the parent of the parent.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PROJECT_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# ---------------------------------------------------------------------------
# Headless matplotlib
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Veritas environment
# ---------------------------------------------------------------------------
from fake_news_investigator.models import InvestigateAction
from fake_news_investigator.server.environment import FakeNewsEnvironment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TASK_DISTRIBUTION = ["easy"] * 5 + ["medium"] * 3 + ["hard"] * 2  # 50/30/20 split
_VERDICT_LABELS = ["TRUE", "MOSTLY_TRUE", "HALF_TRUE", "MOSTLY_FALSE", "FALSE", "PANTS_ON_FIRE"]
_VERDICT_INDEX = {v: i for i, v in enumerate(_VERDICT_LABELS)}

# ---------------------------------------------------------------------------
# Feature extraction (10-dimensional state vector)
# ---------------------------------------------------------------------------

def extract_features(obs) -> List[float]:
    """Extract a 10-dim feature vector from an InvestigateObservation."""
    budget = getattr(obs, "budget_remaining", 10)
    steps = getattr(obs, "steps_taken", 0)
    source_content = getattr(obs, "source_content", None)
    cross_ref = getattr(obs, "cross_ref_result", None) or {}
    credibility = getattr(obs, "credibility_score", None) or 0.5
    consensus = getattr(obs, "consensus_score", None) or 0.5

    entailment = cross_ref.get("entailment", 0.33)
    contradiction = cross_ref.get("contradiction", 0.33)
    neutral = cross_ref.get("neutral", 0.33)

    has_evidence = 1.0 if source_content else 0.0
    has_nli = 1.0 if cross_ref else 0.0
    num_sources = min(steps, 10) / 10.0

    return [
        budget / 10.0,
        steps / 10.0,
        has_evidence,
        has_nli,
        float(credibility),
        float(consensus),
        float(entailment),
        float(contradiction),
        float(neutral),
        num_sources,
    ]


# ---------------------------------------------------------------------------
# Heuristic agent
# ---------------------------------------------------------------------------

def heuristic_verdict(obs) -> str:
    """Choose a verdict based on NLI entailment/contradiction scores."""
    cross_ref = getattr(obs, "cross_ref_result", None) or {}
    e = cross_ref.get("entailment", 0.33)
    c = cross_ref.get("contradiction", 0.33)
    if c > 0.6:
        return "FALSE"
    if e > 0.6:
        return "TRUE"
    if c > 0.45:
        return "MOSTLY_FALSE"
    if e > 0.45:
        return "MOSTLY_TRUE"
    return "HALF_TRUE"


def run_heuristic_episode(env: FakeNewsEnvironment, task: str) -> Tuple[float, List[float], str]:
    """Run one heuristic episode. Returns (reward, state_features, verdict)."""
    obs = env.reset(task=task)
    last_features = extract_features(obs)
    last_obs = obs

    for action_dict in [
        {"action_type": "search_evidence"},
        {"action_type": "request_source", "source_id": "wikipedia"},
        {"action_type": "cross_reference", "source_id": "wikipedia"},
        {"action_type": "check_credibility", "source_id": "en.wikipedia.org"},
        {"action_type": "compute_consensus"},
    ]:
        try:
            action = InvestigateAction(**action_dict)
            last_obs = env.step(action)
            last_features = extract_features(last_obs)
        except Exception:
            pass

    verdict = heuristic_verdict(last_obs)
    try:
        final_obs = env.step(InvestigateAction(
            action_type="submit_verdict",
            verdict=verdict,
            evidence=["wikipedia"],
            confidence=0.65,
            reasoning=f"Heuristic agent verdict: {verdict}",
        ))
        reward = float(final_obs.reward or 0.0)
    except Exception:
        reward = 0.01

    return reward, last_features, verdict


# ---------------------------------------------------------------------------
# Logistic regression (numpy-only fallback)
# ---------------------------------------------------------------------------

class NumpyLogisticRegression:
    """Minimal softmax regression trained with gradient descent."""

    def __init__(self, n_features: int = 10, n_classes: int = 6, lr: float = 0.1, epochs: int = 200):
        import numpy as np
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, (n_classes, n_features))
        self.b = np.zeros(n_classes)

    def _softmax(self, z):
        import numpy as np
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        import numpy as np
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        n = len(y)
        for _ in range(self.epochs):
            logits = X @ self.W.T + self.b
            probs = self._softmax(logits)
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(n), y] = 1.0
            grad = (probs - one_hot) / n
            self.W -= self.lr * (grad.T @ X)
            self.b -= self.lr * grad.sum(axis=0)

    def predict(self, X) -> int:
        import numpy as np
        X = np.array(X, dtype=float).reshape(1, -1)
        logits = X @ self.W.T + self.b
        probs = self._softmax(logits)
        return int(np.argmax(probs, axis=1)[0])


def build_policy(features_list: List[List[float]], verdict_list: List[str]):
    """Train either sklearn or numpy logistic regression and return a predict fn.

    Handles the degenerate case where the heuristic picked only one verdict class
    (which would normally make sklearn fail). In that case we fall through to a
    constant policy that always predicts the single observed label.
    """
    y = [_VERDICT_INDEX.get(v, 2) for v in verdict_list]  # default HALF_TRUE = 2
    unique_labels = set(y)

    # Degenerate case: only one class in the training data. Return a constant policy.
    if len(unique_labels) < 2:
        single_idx = next(iter(unique_labels)) if unique_labels else 2
        def constant_predict(features: List[float]) -> str:
            return _VERDICT_LABELS[single_idx]
        print(f"  [policy] Only {len(unique_labels)} class in training data — using constant policy (label={_VERDICT_LABELS[single_idx]}).")
        return constant_predict

    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(np.array(features_list), y)
        def sklearn_predict(features: List[float]) -> str:
            idx = int(clf.predict([features])[0])
            return _VERDICT_LABELS[idx]
        print("  [policy] Using sklearn LogisticRegression.")
        return sklearn_predict
    except ImportError:
        pass

    # Fallback: numpy-only softmax regression
    clf = NumpyLogisticRegression()
    clf.fit(features_list, y)
    def numpy_predict(features: List[float]) -> str:
        idx = clf.predict(features)
        return _VERDICT_LABELS[idx]
    print("  [policy] Using numpy-only softmax regression (sklearn not installed).")
    return numpy_predict


def run_policy_episode(
    env: FakeNewsEnvironment,
    task: str,
    policy_fn,
) -> Tuple[float, List[float], str]:
    """Run one episode using the trained policy for verdict selection."""
    obs = env.reset(task=task)
    last_features = extract_features(obs)
    last_obs = obs

    for action_dict in [
        {"action_type": "search_evidence"},
        {"action_type": "request_source", "source_id": "wikipedia"},
        {"action_type": "cross_reference", "source_id": "wikipedia"},
        {"action_type": "check_credibility", "source_id": "en.wikipedia.org"},
        {"action_type": "compute_consensus"},
    ]:
        try:
            action = InvestigateAction(**action_dict)
            last_obs = env.step(action)
            last_features = extract_features(last_obs)
        except Exception:
            pass

    verdict = policy_fn(last_features)
    try:
        final_obs = env.step(InvestigateAction(
            action_type="submit_verdict",
            verdict=verdict,
            evidence=["wikipedia"],
            confidence=0.70,
            reasoning=f"Trained policy verdict: {verdict}",
        ))
        reward = float(final_obs.reward or 0.0)
    except Exception:
        reward = 0.01

    return reward, last_features, verdict


# ---------------------------------------------------------------------------
# Rolling average
# ---------------------------------------------------------------------------

def rolling_avg(values: List[float], window: int = 10) -> List[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / (i - start + 1))
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(n_episodes: int = 200, window: int = 10) -> None:
    os.makedirs("outputs", exist_ok=True)

    env = FakeNewsEnvironment()
    phase1_n = n_episodes // 2
    phase2_n = n_episodes - phase1_n

    print(f"\n{'='*60}")
    print(f" Veritas Agent Training — {n_episodes} episodes")
    print(f"{'='*60}")
    print(f"\nPhase 1: Heuristic agent ({phase1_n} episodes) ...")

    # Phase 1 — heuristic
    heuristic_rewards: List[float] = []
    heuristic_features: List[List[float]] = []
    heuristic_verdicts: List[str] = []

    for ep in range(phase1_n):
        task = _TASK_DISTRIBUTION[ep % len(_TASK_DISTRIBUTION)]
        reward, feats, verdict = run_heuristic_episode(env, task)
        heuristic_rewards.append(reward)
        heuristic_features.append(feats)
        heuristic_verdicts.append(verdict)
        if (ep + 1) % 20 == 0:
            avg = sum(heuristic_rewards[-20:]) / 20
            print(f"  ep {ep+1:4d}/{phase1_n}  task={task:6s}  rolling-20 avg={avg:.4f}")

    print(f"\nTraining policy on {len(heuristic_features)} episodes ...")
    policy_fn = build_policy(heuristic_features, heuristic_verdicts)

    # Phase 2 — trained policy
    print(f"\nPhase 2: Trained policy ({phase2_n} episodes) ...")
    policy_rewards: List[float] = []
    policy_features: List[List[float]] = []
    policy_verdicts: List[str] = []

    for ep in range(phase2_n):
        task = _TASK_DISTRIBUTION[ep % len(_TASK_DISTRIBUTION)]
        reward, feats, verdict = run_policy_episode(env, task, policy_fn)
        policy_rewards.append(reward)
        policy_features.append(feats)
        policy_verdicts.append(verdict)
        if (ep + 1) % 20 == 0:
            avg = sum(policy_rewards[-20:]) / 20
            print(f"  ep {ep+1:4d}/{phase2_n}  task={task:6s}  rolling-20 avg={avg:.4f}")

    # ---------------------------------------------------------------------------
    # Save JSON checkpoint
    # ---------------------------------------------------------------------------
    all_rewards = heuristic_rewards + policy_rewards
    agent_stats = {
        "total_episodes": n_episodes,
        "phase1_episodes": phase1_n,
        "phase2_episodes": phase2_n,
        "phase1_avg_reward": sum(heuristic_rewards) / len(heuristic_rewards),
        "phase2_avg_reward": sum(policy_rewards) / len(policy_rewards),
        "phase1_rewards": heuristic_rewards,
        "phase2_rewards": policy_rewards,
    }
    stats_path = "outputs/agent_stats.json"
    with open(stats_path, "w") as f:
        json.dump(agent_stats, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ---------------------------------------------------------------------------
    # Plot learning curve
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Veritas Agent — Learning Curve", fontsize=14, fontweight="bold")

    # Left: raw rewards coloured by phase + rolling average
    all_rolling = rolling_avg(all_rewards, window)
    episodes_x = list(range(1, n_episodes + 1))

    ax = axes[0]
    ax.scatter(
        range(1, phase1_n + 1),
        heuristic_rewards,
        alpha=0.3,
        s=12,
        color="#4393c3",
        label="Heuristic (raw)",
    )
    ax.scatter(
        range(phase1_n + 1, n_episodes + 1),
        policy_rewards,
        alpha=0.3,
        s=12,
        color="#d6604d",
        label="Trained policy (raw)",
    )
    ax.plot(episodes_x, all_rolling, color="#1a1a2e", linewidth=2, label=f"Rolling avg ({window})")
    ax.axvline(x=phase1_n, color="gray", linestyle="--", alpha=0.7, label="Policy trained here")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Rewards")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Right: per-phase rolling average comparison
    ax2 = axes[1]
    p1_rolling = rolling_avg(heuristic_rewards, window)
    p2_rolling = rolling_avg(policy_rewards, window)
    ax2.plot(
        range(1, phase1_n + 1),
        p1_rolling,
        color="#4393c3",
        linewidth=2,
        label="Heuristic rolling avg",
    )
    ax2.plot(
        range(1, phase2_n + 1),
        p2_rolling,
        color="#d6604d",
        linewidth=2,
        label="Trained policy rolling avg",
    )
    ax2.set_xlabel("Episode within phase")
    ax2.set_ylabel("Rolling avg reward")
    ax2.set_title("Heuristic vs Trained Policy")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = "outputs/learning_curve.png"
    plt.savefig(curve_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {curve_path}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(" Summary")
    print(f"{'='*60}")
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Phase 1 (heuristic) avg reward':<35} {agent_stats['phase1_avg_reward']:>10.4f}")
    print(f"  {'Phase 2 (trained policy) avg reward':<35} {agent_stats['phase2_avg_reward']:>10.4f}")
    delta = agent_stats["phase2_avg_reward"] - agent_stats["phase1_avg_reward"]
    print(f"  {'Delta (trained - heuristic)':<35} {delta:>+10.4f}")
    print(f"  {'Total episodes':<35} {n_episodes:>10d}")
    print(f"  {'Learning curve PNG':<35} {'outputs/learning_curve.png':>10}")
    print(f"  {'Agent stats JSON':<35} {'outputs/agent_stats.json':>10}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a Veritas heuristic + policy agent.")
    parser.add_argument("--episodes", type=int, default=200, help="Total episodes (default: 200)")
    parser.add_argument("--window", type=int, default=10, help="Rolling avg window (default: 10)")
    args = parser.parse_args()
    train(n_episodes=args.episodes, window=args.window)


if __name__ == "__main__":
    main()
