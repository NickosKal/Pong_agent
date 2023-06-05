import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env):
    """Performs necessary observation preprocessing."""

    obs = obs/255.0
    return torch.tensor(obs, device=device).float()
