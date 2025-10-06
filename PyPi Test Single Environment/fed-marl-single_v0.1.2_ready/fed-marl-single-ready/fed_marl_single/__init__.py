from .version import __version__  # noqa: F401

try:
    from gymnasium.envs.registration import register
    register(
        id="FederatedSingle-v0",
        entry_point="fed_marl_single.envs.federated_single_env:FederatedSingleEnv",
        max_episode_steps=200,
    )
except Exception:
    # Defer any import issues until actual use
    pass