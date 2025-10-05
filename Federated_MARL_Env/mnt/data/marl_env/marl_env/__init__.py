from gymnasium.envs.registration import register

# Register single-world multi-agent env
register(
    id="FederatedSingle-v0",
    entry_point="marl_env.envs.federated_single_env:FederatedSingleEnv",
)

# Register batched multi-world env
register(
    id="FederatedMulti-v0",
    entry_point="marl_env.envs.federated_multi_env:FederatedMultiEnv",
)
