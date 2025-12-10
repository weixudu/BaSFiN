import json, torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple
from itertools import combinations

# ---------- 共用工具 ----------
def combine(team_size=5):
    index1, index2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            if i == j:
                continue
            index1.append(i)
            index2.append(j)
    return index1, index2

# ---------- 注意力 ----------
class AttM(nn.Module):
    def __init__(self, length=5, hidden_dim=10, reduce=False):
        super().__init__()
        self.length1 = length
        self.length2 = length if not reduce else length - 1
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, team1, team2, dim=2):
        assert team1.shape == team2.shape
        l1, l2 = self.length1, self.length2
        team1 = team1.view(-1, l1, l2, self.hidden_dim)
        team2 = team2.view(-1, l1, l2, self.hidden_dim)
        score = (self.W(team1) * team2).sum(dim=3)          # [B,5,4]
        score = F.softmax(score, dim=dim)                   # [B,5,4]
        return score.view(-1, l1 * l2)                      # [B,20]

# ---------- ANFM（舊版邏輯 + pairwise 支援） ----------
class ANFM(nn.Module):
    def __init__(self, n_player, player_dim, team_size=5,
                 hidden_dim=10, need_att=False,
                 mlp_hidden_dim=50, dropout_rate=0.2):
        super().__init__()
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.need_att  = need_att

        self.embedding = nn.Embedding(n_player, player_dim)
        self.feature_mapper = nn.Sequential(
            nn.Linear(33 + player_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

        self.index1, self.index2 = combine(team_size)
        self.attenM = AttM(team_size, hidden_dim, reduce=True)  # 舊版：永遠建構

        dropout = nn.Dropout(dropout_rate, inplace=False)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            dropout,
            nn.Linear(mlp_hidden_dim, 1, bias=True)
        )
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, team_feats, team_ids, return_pairwise=False):
        # team_feats: [B,5,33]  team_ids: [B,5]
        team_emb = self.embedding(team_ids)                           # [B,5,P]
        x = torch.cat((team_feats, team_emb), dim=-1)                 # [B,5,33+P]
        h = self.feature_mapper(x)                                    # [B,5,H]

        a = h[:, self.index1]                                         # [B,20,H]
        b = h[:, self.index2]                                         # [B,20,H]
        order2 = self.MLP(a * b).squeeze(-1)                          # [B,20]

        if self.need_att:                                             # 與舊版一致
            normal = self.attenM(a, b, dim=2)
            order2 = order2.mul(normal)

        if return_pairwise:                                           # 新增功能
            return order2                                             # [B,20]

        return order2.sum(dim=1, keepdim=True)                        # [B,1]

# ---------- NAC-ANFM（舊版邏輯 + pairwise 支援） ----------
class NAC_ANFM(nn.Module):
    def __init__(self, n_player, player_dim,
                 team_size=5, hidden_dim=10, need_att=False,
                 mlp_hidden_dim=50, dropout_rate=0.2,
                 device=torch.device('cpu'),
                 ema_tensor_path=None, game_id_mapping_path=None):
        super().__init__()
        self.team_size = team_size
        self.device    = device

        self.Coop = ANFM(n_player, player_dim, team_size,
                         hidden_dim, need_att,
                         mlp_hidden_dim, dropout_rate).to(device)

        if ema_tensor_path is None or game_id_mapping_path is None:
            raise ValueError("ema_tensor_path & game_id_mapping_path 必須提供")
        self.game_ema_tensor = torch.load(ema_tensor_path, weights_only=True).to(device)

        with open(game_id_mapping_path, 'r', encoding='utf-8') as f:
            m = json.load(f)
        self.game_id_to_index = {v: int(k) for k, v in m.items()}

    # —— 舊版的 feature loading —— #
    def load_ema_features(self, game_ids, noise_scale=0.01):
        # game_ids: [B]
        game_indices = torch.tensor(
            [self.game_id_to_index.get(str(int(gid.item())), -1) for gid in game_ids],
            dtype=torch.long, device=self.device
        )
        invalid = game_indices.eq(-1)
        if invalid.any():
            print(f"Warning: {invalid.sum().item()} invalid game_ids")
            game_indices[invalid] = 0

        feats = self.game_ema_tensor[game_indices]        # [B,10,33]
        return feats[:, :5, :], feats[:, 5:, :]           # team_A, team_B

    # —— forward：保留 pairwise 功能 —— #
    def forward(self, data, need_pairwise=False):
        # data: [B, 1+10] → game_id + 10 players
        data   = torch.as_tensor(data, dtype=torch.long, device=self.device)
        g_ids  = data[:, 0]
        p_ids  = data[:, 1:]
        A_ids, B_ids = p_ids[:, :self.team_size], p_ids[:, self.team_size:]

        A_f, B_f = self.load_ema_features(g_ids)

        if need_pairwise:
            pair_A = self.Coop(A_f, A_ids, return_pairwise=True)   # [B,20]
            pair_B = self.Coop(B_f, B_ids, return_pairwise=True)   # [B,20]
            score_A = pair_A.sum(1, keepdim=True)                  # [B,1]
            score_B = pair_B.sum(1, keepdim=True)
        else:
            score_A = self.Coop(A_f, A_ids)                        # [B,1]
            score_B = self.Coop(B_f, B_ids)

        prob = torch.sigmoid(score_A - score_B).squeeze(-1)        # [B]

        if need_pairwise:
            return prob, score_A, score_B, pair_A, pair_B
        return prob, score_A, score_B
