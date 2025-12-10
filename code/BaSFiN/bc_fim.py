import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ----------------------------- 共用工具 ------------------------------

def combine(team_size: int = 5):
    """建立 5×5 → 25 的索引 (包含 i==j)。"""
    index1, index2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            index1.append(i)
            index2.append(j)
    return index1, index2

# ------------------------------- AttM ------------------------------

class AttM(nn.Module):
    """沿用舊版注意力 (固定 dim=2 softmax)。"""

    def __init__(self, length: int = 5, hidden_dim: int = 10, reduce: bool = False):
        super().__init__()
        self.length1 = length
        self.length2 = length if not reduce else length - 1
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, team1: torch.Tensor, team2: torch.Tensor):
        # team1/team2: [B*25, H]  (flatten 後)
        assert team1.shape == team2.shape
        l1, l2 = self.length1, self.length2
        team1 = team1.view(-1, l1, l2, self.hidden_dim)
        team2 = team2.view(-1, l1, l2, self.hidden_dim)
        score = (self.W(team1) * team2).sum(dim=3)  # [B,5,5]
        score = F.softmax(score, dim=2)
        return score.view(-1, l1 * l2)              # [B,25]

# ------------------------ FeatureInteraction -----------------------

class FeatureInteraction(nn.Module):
    """
    沿用【舊版】特徵互動邏輯，僅新增 `return_pairwise` 以支援回傳 25 個配對分數。
    """

    def __init__(
        self,
        n_player: int,
        player_dim: int,
        feature_dim: int = 33,
        context_dim: int = 1,
        intermediate_dim: int = 16,
        team_size: int = 5,
        mlp_hidden_dim: int = 50,
        dropout_rate: float = 0.2,
        need_att: bool = True,
    ) -> None:
        super().__init__()
        self.team_size = team_size
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.intermediate_dim = intermediate_dim
        self.need_att = need_att
        self.index1, self.index2 = combine(team_size)

        # 嵌入層
        self.embedding = nn.Embedding(n_player, player_dim)

        input_dim = player_dim + feature_dim + context_dim
        self.blade_emb = nn.Sequential(
            nn.Linear(input_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, intermediate_dim),
        )
        self.chest_emb = nn.Sequential(
            nn.Linear(input_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, intermediate_dim),
        )

        self.mlp_pairwise = nn.Sequential(
            nn.Linear(intermediate_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, 1, bias=True),
        )

        if need_att:
            self.attenM = AttM(length=team_size, hidden_dim=intermediate_dim, reduce=False)

        self.apply(self._init)

    # ------------------- forward -------------------

    def forward(
        self,
        team_A_ids: torch.Tensor,
        team_B_ids: torch.Tensor,
        feats_A: torch.Tensor,
        feats_B: torch.Tensor,
        z_g_A: torch.Tensor,
        z_g_B: torch.Tensor,
        return_pairwise: bool = False,
    ) -> torch.Tensor:
        """若 `return_pairwise=True` → 回傳 [B,25]；否則 [B,1]。"""
        B = feats_A.shape[0]

        # 嵌入 & concat
        emb_A = self.embedding(team_A_ids)                          # [B,5,P]
        emb_B = self.embedding(team_B_ids)

        zA = z_g_A.unsqueeze(-1).unsqueeze(-1).expand(-1, self.team_size, 1)
        zB = z_g_B.unsqueeze(-1).unsqueeze(-1).expand(-1, self.team_size, 1)

        cat_A = torch.cat([emb_A, feats_A, zA], dim=-1)             # [B,5,input]
        cat_B = torch.cat([emb_B, feats_B, zB], dim=-1)

        a_feat = self.blade_emb(cat_A)                              # [B,5,H]
        b_feat = self.chest_emb(cat_B)

        a_pair = a_feat[:, self.index1, :]                          # [B,25,H]
        b_pair = b_feat[:, self.index2, :]

        interact = a_pair * b_pair                                  # [B,25,H]
        score = self.mlp_pairwise(interact).squeeze(-1)             # [B,25]

        if self.need_att:
            normal = self.attenM(a_pair, b_pair)                    # [B,25]
            score = score * normal

        if return_pairwise:
            return score                                            # [B,25]
        else:
            return score.sum(dim=1, keepdim=True)                  # [B,1]

    # ---------------- weight init -----------------

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)

# ------------------------------ FIModel -----------------------------

class FIModel(nn.Module):
    """沿用舊版外層結構，加入 `need_pairwise` 輸出。"""

    def __init__(
        self,
        n_player: int,
        player_dim: int,
        feature_dim: int = 33,
        context_dim: int = 1,
        intermediate_dim: int = 16,
        team_size: int = 5,
        mlp_hidden_dim: int = 50,
        dropout_rate: float = 0.2,
        device: torch.device = torch.device("cpu"),
        ema_tensor_path: str | None = None,
        game_id_mapping_path: str | None = None,
        need_att: bool = True,
    ) -> None:
        super().__init__()
        self.team_size = team_size
        self.device = device
        self.feature_dim = feature_dim

        self.interaction = FeatureInteraction(
            n_player=n_player,
            player_dim=player_dim,
            feature_dim=feature_dim,
            context_dim=context_dim,
            intermediate_dim=intermediate_dim,
            team_size=team_size,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout_rate=dropout_rate,
            need_att=need_att,
        ).to(device)

        # EMA 讀檔 (舊版邏輯)
        if ema_tensor_path is None or game_id_mapping_path is None:
            raise ValueError("ema_tensor_path & game_id_mapping_path 必須提供")
        self.game_ema_tensor = torch.load(ema_tensor_path, weights_only=True).to(device)
        with open(game_id_mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.game_id_to_index = {v: int(k) for k, v in mapping.items()}

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_ema_features(self, game_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.tensor([
            self.game_id_to_index.get(str(int(g.item())), -1) for g in game_ids
        ], dtype=torch.long, device=self.device)
        invalid = idx.eq(-1)
        if invalid.any():
            print(f"Warning: {invalid.sum()} invalid game_id(s); fallback to index 0")
            idx[invalid] = 0
        feats = self.game_ema_tensor[idx]                      # [B,10,?]
        feats_A = feats[:, : self.team_size, : self.feature_dim].clone()
        feats_B = feats[:, self.team_size :, : self.feature_dim].clone()
        return feats_A, feats_B

    # -------------- forward ----------------

    def forward(self, data: torch.Tensor, need_pairwise: bool = False):
        """輸入 `data=[B,1+10]`；根據 need_pairwise 輸出不同格式。"""
        B = data.shape[0]
        data = torch.as_tensor(data, dtype=torch.long, device=self.device)

        # context scalar
        z_off   = torch.ones(B, device=self.device)
        z_def   = torch.zeros(B, device=self.device)
        z_off_o = torch.zeros(B, device=self.device)
        z_def_o = torch.ones(B, device=self.device)

        # split ids
        g_ids = data[:, 0]
        p_ids = data[:, 1:]
        A_ids, B_ids = p_ids[:, : self.team_size], p_ids[:, self.team_size:]

        feats_A, feats_B = self.load_ema_features(g_ids)

        # 主攻 vs 客守
        out1 = self.interaction(A_ids, B_ids, feats_A, feats_B, z_off, z_def, return_pairwise=need_pairwise)
        # 客攻 vs 主守
        out2 = self.interaction(B_ids, A_ids, feats_B, feats_A, z_off_o, z_def_o, return_pairwise=need_pairwise)

        if need_pairwise:
            pair1, pair2 = out1, out2
            v_score     = pair1.sum(dim=1, keepdim=True)
            v_opp_score = pair2.sum(dim=1, keepdim=True)
            probs       = torch.sigmoid(v_score - v_opp_score).squeeze(-1)
            return probs, v_score, v_opp_score, pair1, pair2
        else:
            v_score, v_opp_score = out1, out2                   # already [B,1]
            probs = torch.sigmoid(v_score - v_opp_score).squeeze(-1)
            return probs, v_score, v_opp_score
