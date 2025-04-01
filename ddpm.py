import torch
import math

# 噪声以线性方式增长
def linear_beta_schedule(timesteps):
    scale = 1000/timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# 噪声以余弦方式增长
def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linespace(0, timesteps, steps, dtype = torch.float64)
    alpha_cumprod = torch.cos(((x/timesteps)+s)/(1+s)*math.pi*0.5)**2
    alpha_cumprod = alpha_cumprod/alpha_cumprod[0]
    betas = 1-(alpha_cumprod[1:]/alpha_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
