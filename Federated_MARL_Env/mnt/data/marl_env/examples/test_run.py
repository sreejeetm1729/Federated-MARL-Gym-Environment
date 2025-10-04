import gymnasium as gym

def smoke_single():
    env = gym.make("FederatedSingle-v0", render_mode="rgb_array", grid_size=8, num_agents=3, max_steps=30)
    obs, info = env.reset()
    total = 0.0
    for _ in range(30):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    arr = env.render()  # rgb array
    env.close()
    print("Single OK. Reward:", total, "Obs shape:", obs.shape, "Frame:", None if arr is None else arr.shape)

def smoke_multi():
    env = gym.make("FederatedMulti-v0", render_mode="rgb_array", worlds=3, grid_size=8, num_agents=3, max_steps=20)
    obs, info = env.reset()
    total = 0.0
    for _ in range(20):
        a = tuple(env.action_space.spaces[i].sample() for i in range(env.cfg.worlds))
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    arr = env.render()
    env.close()
    print("Multi OK. Reward:", total, "Obs tuple len:", len(obs), "Frame:", None if arr is None else arr.shape)

if __name__ == "__main__":
    smoke_single()
    smoke_multi()
