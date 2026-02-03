import torch
import torch.nn as nn


# ==========================================
# 1. Baseline: Discrete Jump Model
# ==========================================
class JumpModel(nn.Module):
    """
    Learns x_{t+1} = x_t + f(x_t, a_t)
    Approximates dynamics as a discrete map (Chord).
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

    def forward(self, state, action):
        emb = self.act_emb(action)
        delta = self.net(torch.cat([state, emb], dim=-1))
        return state + delta


class LSTMModel(nn.Module):
    def __init__(self, state_dim=2, action_dim=3, action_embedding_dim=8, hidden_dim=64):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)
        self.lstm = nn.LSTM(state_dim + action_embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        emb = self.act_emb(action)
        x = torch.cat([state, emb], dim=-1)
        x, _ = self.lstm(x)
        return state + self.fc(x)


class NewtonianDynamicsModel(nn.Module):
    """
    Learns d/dt[x, v] = [v, f(x, v, a)]
    """

    def __init__(self, state_dim=2, action_dim=3, action_embedding_dim=8, hidden_dim=64, damping_init=-1.0):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # Output: Acceleration 
        )

        self.log_damping = nn.Parameter(torch.tensor(damping_init))

    def forward(self, t, state, action_emb):
        x, v = state[..., 0:1], state[..., 1:2]

        # learned damping
        damping = torch.exp(self.log_damping)
        acc_pred = self.net(torch.cat([x, v, action_emb], dim=-1))
        forces = acc_pred - damping * v

        return torch.cat([v, forces], dim=-1)


class VelocityDynamicsModel(nn.Module):
    """
    Learns d/dt[x] = Prior(a) + NN(x, a)
    Velocity Dynamics Model: predicts velocity directly
    """

    def __init__(self, state_dim=2, action_dim=3, action_embedding_dim=8, hidden_dim=64):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        # Neural Correction Term
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1), 
        )

    def forward(self, t, state, action_emb):
        vel = self.net(torch.cat([state, action_emb], dim=-1))
        return torch.cat([vel, torch.zeros_like(vel)], dim=-1) # dxdt,dvdt: zeros for dvdt so we can use the same integrator as the NewtonianDynamicsModel


class PortHamiltonianModel(nn.Module):
    def __init__(self, state_dim=2, action_dim=3, action_embedding_dim=8, hidden_dim=64, damping_init=-1.0):
        super().__init__()
        self.act_emb = nn.Embedding(action_dim, action_embedding_dim)

        self.H_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )

        self.log_damping = nn.Parameter(torch.tensor(damping_init))

        # 4. Input Port (G)
        self.G_net = nn.Linear(action_embedding_dim, state_dim // 2)

    def forward(self, t, state, action_emb):
        if not state.requires_grad:
            state.requires_grad_(True)

        with torch.enable_grad():
            H = self.H_net(state)
            dH_dstate = torch.autograd.grad(
                H, state, torch.ones_like(H), create_graph=True
            )[0]
            dH_dq, dH_dp = dH_dstate[..., 0:1], dH_dstate[..., 1:2]

        # dissipation
        damping = torch.exp(self.log_damping)

        # external force
        G_u = self.G_net(action_emb) 

        # dynamics
        dq_dt = dH_dp
        dp_dt = -dH_dq - (damping * dH_dp) + G_u

        return torch.cat([dq_dt, dp_dt], dim=-1)
