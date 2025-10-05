# Federated MARL Gym Environment

Two small, self-contained grid environments for testing **federated / multi-agent RL**:

- `FederatedSingle-v0`: One world with multiple agents (drones) and moving building goals.
- `FederatedMulti-v0`: Multiple worlds in parallel (batched), useful for federated/server aggregation loops.

## Install

```bash
pip install -e .
```

## Quick test

```python
import gymnasium as gym
env = gym.make("FederatedSingle-v0", render_mode="human", grid_size=12, num_agents=5)
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    env.render()
    if term or trunc:
        obs, info = env.reset()
env.close()
```

For a non-window pipeline, pass `render_mode="rgb_array"` and collect frames.

## Assets

If `assets/drone.png` or `assets/building.png` are missing, placeholders are auto-generated. To use your own, place them under `marl_env/assets/`.

## Notes

- API follows Gymnasium: `reset()` returns `(obs, info)`, `step()` returns `(obs, reward, terminated, truncated, info)`.
- Action space is `spaces.MultiDiscrete([5]*num_agents)` per world: `0=stay, 1=up, 2=down, 3=left, 4=right`.
