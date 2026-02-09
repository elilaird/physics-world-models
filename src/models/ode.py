import torch
import torch.nn as nn


class FirstOrderODENet(nn.Module):
    """
    Learns d/dt[state] = NN(state, action_emb)
    First-order ODE: predicts full state derivative directly.
    """

    def __init__(self, state_dim=2, action_dim=3, action_embedding_dim=8, hidden_dim=64):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, t, state, action_emb):
        nn_input = torch.cat([state, action_emb], dim=-1)
        return self.net(nn_input)


class NewtonianDynamicsModel(nn.Module):
    """
    Learns d/dt[x, v] = [v, f(x, v, a)]
    Separates position/velocity with learned damping.
    """

    def __init__(self, state_dim=2, action_dim=3, action_embedding_dim=8, hidden_dim=64, damping_init=-1.0):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_damping = nn.Parameter(torch.tensor(damping_init))

    def forward(self, t, state, action_emb):
        x, v = state[..., 0:1], state[..., 1:2]
        damping = torch.exp(self.log_damping)
        acc_pred = self.net(torch.cat([x, v, action_emb], dim=-1))
        forces = acc_pred - damping * v
        return torch.cat([v, forces], dim=-1)


class VelocityDynamicsModel(nn.Module):
    """
    Learns d/dt[x] = NN(x, a), with zeros for dvdt.
    Shares ODE integrator interface with NewtonianDynamicsModel.
    """

    def __init__(self, state_dim=2, action_dim=3, action_embedding_dim=8, hidden_dim=64):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t, state, action_emb):
        vel = self.net(torch.cat([state, action_emb], dim=-1))
        return torch.cat([vel, torch.zeros_like(vel)], dim=-1)
