import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import json

def combine(team_size=5):
    index1, index2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            if i == j:
                continue
            index1.append(i)
            index2.append(j)
    return index1, index2

class ANFM(nn.Module):
    def __init__(self, n_player, player_dim, team_size, hidden_dim, need_att=False, mlp_hidden_dim=50, dropout_rate=0.2):
        super(ANFM, self).__init__()
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_player, player_dim)
        self.feature_mapper = nn.Sequential(
            nn.Linear(33 + player_dim, 2 * hidden_dim),  
            nn.SiLU(),                                  
            nn.Linear(2 * hidden_dim, hidden_dim)       
        )
        self.index1, self.index2 = combine(team_size)
        self.need_att = need_att
        self.attenM = AttM(team_size, hidden_dim, reduce=True)
        dropout = nn.Dropout(dropout_rate, inplace=False)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            dropout,
            nn.Linear(mlp_hidden_dim, 1, bias=True),
        )
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, team_features, team_ids):
        # team_features: [batch_size, team_size, 32]  # Changed from 37 to 32
        # team_ids: [batch_size, team_size]
        n_match = len(team_features)
        team_emb = self.embedding(team_ids)  # [batch_size, team_size, player_dim]
        concat_features = torch.cat((team_features, team_emb), dim=-1)  # [batch_size, team_size, 32 + player_dim] # [batch_size, team_size, 32 + player_dim]  # Changed from 37 to 32
        embedded = self.feature_mapper(concat_features)  # [batch_size, team_size, hidden_dim]
        a = embedded[:, self.index1]  # [batch_size, team_size * (team_size - 1), hidden_dim]
        b = embedded[:, self.index2]  # [batch_size, team_size * (team_size - 1), hidden_dim]
        order2 = self.MLP(a * b).squeeze(-1)  # [batch_size, team_size * (team_size - 1)]
        if self.need_att:
            normal = self.attenM(a, b, dim=2)
            order2 = order2.mul(normal)
        order2 = order2.sum(dim=1, keepdim=True)  # [batch_size, 1]
        return order2

class AttM(nn.Module):
    def __init__(self, length=5, hidden_dim=10, reduce=False):
        super(AttM, self).__init__()
        self.length1 = length
        self.length2 = length if not reduce else length - 1
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, team1, team2, dim=2):
        assert team1.shape == team2.shape
        length1, length2 = self.length1, self.length2
        team1 = team1.view(-1, length1, length2, self.hidden_dim)
        team2 = team2.view(-1, length1, length2, self.hidden_dim)
        score = (self.W(team1) * team2).sum(dim=3)  # [batch_size, 5, 4]
        score = F.softmax(score, dim=dim)  # [batch_size, 5, 4]
        return score.view(-1, length1 * length2)

class NAC_ANFM(nn.Module):
    def __init__(self, n_player, player_dim, team_size=5, hidden_dim=10, need_att=False, mlp_hidden_dim=50, dropout_rate=0.2,
                 device=torch.device('cpu'), ema_tensor_path=None, game_id_mapping_path=None):
        super(NAC_ANFM, self).__init__()
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.Coop = ANFM(n_player, player_dim, team_size, hidden_dim, need_att, mlp_hidden_dim, dropout_rate)
        # Load game_ema_tensor and game_id_mapping
        if ema_tensor_path is None or game_id_mapping_path is None:
            raise ValueError("ema_tensor_path and game_id_mapping_path must be provided")
        self.game_ema_tensor = torch.load(ema_tensor_path, weights_only=True).to(self.device)
        with open(game_id_mapping_path, 'r', encoding='utf-8') as f:
            self.game_id_mapping = json.load(f)
        self.game_id_to_index = {v: int(k) for k, v in self.game_id_mapping.items()}
        
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def load_ema_features(self, data, game_ids, noise_scale=0.01):
        batch_size = data.shape[0]
        # Map game_ids to tensor indices
        game_indices = torch.tensor([self.game_id_to_index.get(str(int(gid.item())), -1) for gid in game_ids],
                                   dtype=torch.long, device=self.device)
        invalid_indices = game_indices == -1
        if invalid_indices.any():
            print(f"Warning: {invalid_indices.sum()} invalid game_ids")
            game_indices[invalid_indices] = 0  # Fill invalid indices with 0

        # Extract features from game_ema_tensor
        features = self.game_ema_tensor[game_indices]  # [batch_size, 10, 32]  # Changed from 37 to 32
        
        # Split into team_A and team_B
        team_A_features = features[:, :5, :]  # [batch_size, 5, 32]  # Changed from 37 to 32
        team_B_features = features[:, 5:, :]  # [batch_size, 5, 32]  # Changed from 37 to 32

        # # Add noise for game_id <= 30
        # mask = (game_ids <= 30).view(-1, 1, 1).to(self.device)
        # if mask.any():
        #     team_A_features = team_A_features + mask * torch.randn_like(team_A_features) * noise_scale
        #     team_B_features = team_B_features + mask * torch.randn_like(team_B_features) * noise_scale

        return team_A_features, team_B_features

    def forward(self, data):
        data = torch.LongTensor(data).to(self.device)
        # Extract game_ids and player_data
        game_ids = data[:, 0]
        player_data = data[:, 1:]
        team_A_ids = player_data[:, :self.team_size]  # [batch_size, team_size]
        team_B_ids = player_data[:, self.team_size:]  # [batch_size, team_size]
        team_A_features, team_B_features = self.load_ema_features(player_data, game_ids)
        A_coop = self.Coop(team_A_features, team_A_ids)  # [batch_size, 1]
        B_coop = self.Coop(team_B_features, team_B_ids)  # [batch_size, 1]

        probs = torch.sigmoid(A_coop - B_coop).squeeze(-1)  # [batch_size]
        return probs, A_coop, B_coop