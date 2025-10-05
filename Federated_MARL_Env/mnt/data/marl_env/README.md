# marl-env: Federated Multi‑Agent RL Gymnasium Environments

> **Federated MARL** environments (single & multi‑world variants) built with **Gymnasium** + **Pygame**, designed to showcase collaborative speedups and evaluate federated / decentralized RL algorithms (Q‑learning, robust Q‑learning, Fed‑Q, etc.).

- ✅ **Gymnasium‑compatible** IDs: `FederatedSingle-v0`, `FederatedMulti-v0`
- 🛰️ **Agents = drones**, **goals = buildings** (sprites included / auto‑generated placeholders)
- 🖼️ **Two render modes**: `human` (Pygame window) and `rgb_array` (numpy frames)
- 🎥 **Optional recording** helpers (MP4/GIF) via `opencv-python` or `imageio` + `imageio-ffmpeg`
- 🧪 Great for testing **federated / decentralized** algorithms, collaborative speedups, and robustness

---

## 1) Install (TestPyPI)

```bash
# create a fresh environment (recommended)
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate

# install from TestPyPI (note the -i flag)
pip install -i https://test.pypi.org/simple/ marl-env
# if pillow / pygame fail to build wheels on your platform, try prebuilt wheels:
pip install -i https://test.pypi.org/simple/ marl-env pillow pygame
```

> When you promote to real PyPI later, users can simply do: `pip install marl-env` (no `-i`).

---

## 2) Quickstart

### A) Minimal smoke test (no training)
```python
import gymnasium as gym
import marl_env   # registers env IDs on import

env = gym.make("FederatedSingle-v0", render_mode="rgb_array", grid_size=8, num_agents=3)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

### B) Pygame window (interactive)
```python
import os
# ensure we DO NOT force headless mode
if os.environ.get("SDL_VIDEODRIVER") == "dummy":
    del os.environ["SDL_VIDEODRIVER"]

import gymnasium as gym, marl_env
env = gym.make("FederatedSingle-v0", render_mode="human", grid_size=10, num_agents=5, fps=30)
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    done = term or trunc
env.close()
```

### C) Multi‑world variant
```python
import gymnasium as gym, marl_env

env = gym.make("FederatedMulti-v0",
               render_mode="rgb_array",
               worlds=3, cols=3,    # world layout
               grid_size=10,
               num_agents=5,
               fps=20)
obs, info = env.reset()
for _ in range(50):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc: break
env.close()
```

---

## 3) Environments Overview

### `FederatedSingle-v0`
- Single grid world with `num_agents` drones delivering to building goals
- **Action space**: `MultiDiscrete([5] * num_agents)` (0: stay, 1: up, 2: down, 3: left, 4: right)
- **Observation**: concatenated and normalized `(agent_xy, goal_xy)` for all agents
- **Rewards**: +1 per agent on delivery, small step penalty (e.g., −0.01) to encourage efficiency
- **Episode end**: fixed `max_steps` or all agents deliver (configurable)

### `FederatedMulti-v0`
- Row/column layout of **heterogeneous worlds** (e.g., different obstacle maps or step costs)
- Optional **central broadcast** cell for illustrating server‑style coordination
- Same action/observation shape per world; stacked into a composite environment for federated RL

> **Sprites**: If `assets/drone.png` or `assets/building.png` are missing, placeholder icons are auto‑generated.

---

## 4) Jupyter vs. Desktop (Pygame)

- **Notebook rendering**: set `render_mode="rgb_array"` and display the returned frame arrays.
- **Desktop window**: set `render_mode="human"`. If you previously set headless mode for Pygame (`SDL_VIDEODRIVER=dummy`), make sure to unset it before creating the env (see Quickstart B).

---

## 5) Recording (optional)

Install recording extras:
```bash
pip install marl-env[record]
```

Example (MP4 preferred; falls back to GIF):
```python
import numpy as np, imageio
frames = []
obs, info = env.reset()
for _ in range(200):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    frame = env.render()  # when render_mode="rgb_array"
    if frame is not None:
        frames.append(frame)
    if term or trunc: break

# Write video
imageio.mimsave("rollout.gif", frames, fps=20)   # or use imageio-ffmpeg for MP4
```

---

## 6) API (typical kwargs)

Both IDs accept common kwargs; notable ones:
- `grid_size: int` (default 10)
- `num_agents: int` (default 5)
- `max_steps: int` (default 500)
- `fps: int` (for human render; default 30)
- `render_mode: Literal["human", "rgb_array"]`

`FederatedMulti-v0` adds:
- `worlds: int` (default 3) — number of worlds (rows × 1 by default; use `cols`)
- `cols: int` (default 3) — arrange worlds in a grid
- Optional heterogeneity flags (maps, noise, penalties) depending on your implementation

> Exact kwargs are enforced by your underlying env classes. See docstrings in `marl_env/*_env.py` once you copy your implementations in.

---

## 7) Troubleshooting

- **`NameNotFound: Cannot find environment`** → Ensure `import marl_env` occurs **before** `gym.make(...)`. The import triggers registration.
- **Pygame window not appearing** → Unset `SDL_VIDEODRIVER=dummy`; check you’re not in a headless/SSH session.
- **TestPyPI install works but import fails** → Confirm the env classes are included in the built wheel. See packaging layout below.
- **`TiffWriter.write() got an unexpected keyword argument 'fps'`** → That’s from Pillow’s GIF writer. Prefer MP4 via `imageio-ffmpeg` or avoid passing `fps` to the Pillow writer.

---

## 8) Project Layout (what this package expects)

```
marl-env/
├── pyproject.toml
├── README.md
├── LICENSE
├── MANIFEST.in
└── marl_env/
    ├── __init__.py                # registers Gymnasium env IDs
    ├── single_env.py              # your FederatedSingleEnv implementation
    ├── multi_env.py               # your FederatedMultiEnv implementation
    ├── assets/
    │   ├── drone.png              # optional (auto-generated if missing)
    │   └── building.png           # optional
    └── utils.py                   # optional recording / sprite helpers
```

> **You have two notebooks**; just copy the actual env class code into `single_env.py` and `multi_env.py`, then import & register them in `__init__.py` (already scaffolded).

---

## 9) Contributing

- Style: `ruff` + `black` (optional)
- PRs: please include a minimal training script or smoke test
- Provide deterministic seeds for unit tests where feasible

---

## 10) License & Citation

**License:** MIT (see `LICENSE`).

**If you use this in research, please cite:**

```bibtex
@misc{marl_env_2025,
  title        = {marl-env: Federated Multi-Agent RL Gymnasium Environments},
  author       = {Maity, Sreejeet},
  howpublished = {TestPyPI package},
  year         = {2025},
  note         = {https://test.pypi.org/project/marl-env/}
}
```

---

## 11) Maintainers

- **Sreejeet Maity** (NCSU) — federated & robust RL, multi‑agent systems

---

## 12) Build & Upload (for you, the maintainer)

From the project root (the folder containing `pyproject.toml`):
```bash
python -m pip install --upgrade build twine

# clean builds
rm -rf dist build *.egg-info

# build wheel + sdist
python -m build

# upload to TestPyPI using your token (project: marl-env)
python -m twine upload --repository testpypi dist/*
# Username: __token__
# Password: pypi-<YOUR TESTPYPI TOKEN>
```

Then verify in a fresh environment:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -i https://test.pypi.org/simple/ marl-env
python -c "import gymnasium as gym, marl_env; print(gym.make('FederatedSingle-v0'))"
```
