import torch
import torch.nn as nn


class PortHamiltonianModel(nn.Module):
    """
    Learns Hamiltonian H(q, p) and derives symplectic dynamics via autograd.
    Includes dissipation (learned damping) and input port G(u) for external forces.
    """

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

        damping = torch.exp(self.log_damping)
        G_u = self.G_net(action_emb)

        dq_dt = dH_dp
        dp_dt = -dH_dq - (damping * dH_dp) + G_u

        return torch.cat([dq_dt, dp_dt], dim=-1)
