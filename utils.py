def normalize_obs(obs):
    return obs / (1 + abs(obs).max())
