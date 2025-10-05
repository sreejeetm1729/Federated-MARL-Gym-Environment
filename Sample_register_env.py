from gymnasium.envs.registration import register

register(
    id="FederatedSingle-v0",
    entry_point="federated_single_env:FederatedSingleEnv",
)
