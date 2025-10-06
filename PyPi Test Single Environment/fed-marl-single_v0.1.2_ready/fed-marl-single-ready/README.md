# fed-marl-single

A tiny, Gymnasium-compatible *single* environment scaffold called **FederatedSingleEnv**.
It registers under the id: `FederatedSingle-v0` upon import. Supports variable numbers of agents and flexible grid dimensions.
> Replace the placeholder dynamics with your own environment (see `fed_marl_single/envs/federated_single_env.py`).

## Install from source

```bash
pip install -U build twine
pip install -e .
python -c "import gymnasium as gym, fed_marl_single; env=gym.make('FederatedSingle-v0'); print(env.reset()[0].shape); env.close()"
```

## Upload to TestPyPI

```bash
python -m build
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# Username: __token__
# Password: pypi-<your TestPyPI token>
```

## Try install from TestPyPI

```bash
pip install -i https://test.pypi.org/simple/ fed-marl-single
python -c "import gymnasium as gym, fed_marl_single; env=gym.make('FederatedSingle-v0'); env.reset(); env.step(env.action_space.sample()); print('ok'); env.close()"
```

## Example usage

```python
import gymnasium as gym
import fed_marl_single  # registers envs

env = gym.make("FederatedSingle-v0", render_mode="rgb_array", grid_size=8, num_agents=3)
obs, info = env.reset()
for _ in range(10):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
env.close()
```
