
# fed-marl-multi (v0.2.2)

Federated multi-environment visualization with **Broadcaster** panel (left) and a single row of **distinctly colored** client boards (right).
Each environment gets its own color theme for tiles, goal tint, and agent halos.

- Variable `num_agents`, `num_envs`
- Actions: `(num_agents,)` broadcast or `(num_envs, num_agents)` per-board
- Render modes: `"human"` (pygame window) and `"rgb_array"` (Jupyter-safe)

## Quick use
```python
import gymnasium as gym, fed_marl_multi, numpy as np

env = gym.make("FederatedMulti-v0",
               grid_size=10,
               num_agents=3,
               num_envs=6,
               render_mode="rgb_array")
obs, info = env.reset()
frame = env.render()
print(info, frame.shape)
env.close()
```
