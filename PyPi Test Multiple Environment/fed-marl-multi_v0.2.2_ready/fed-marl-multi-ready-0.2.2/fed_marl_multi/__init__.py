
from .version import __version__  # noqa: F401
try:
    from gymnasium.envs.registration import register
    register(
        id="FederatedMulti-v0",
        entry_point="fed_marl_multi.envs.federated_multi_env:FederatedMultiEnv",
        max_episode_steps=300,
    )
except Exception:
    pass
