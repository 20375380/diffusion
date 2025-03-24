def linear_beta_schedule(timesteps):
    scale = 1000/timesteps
    bate_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
