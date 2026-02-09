import torch
import torch.nn as nn


class JumpModel(nn.Module):
    """
    Learns x_{t+1} = x_t + f(x_t, a_t)
    Approximates dynamics as a discrete map (residual MLP).
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
    """
    LSTM with residual connection for sequential prediction.
    """

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
