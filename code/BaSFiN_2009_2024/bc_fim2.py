import torch
import torch.nn as nn
import json
import torch.nn.functional as F

def combine(team_size=5):
    index1, index2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            index1.append(i)
            index2.append(j)
    return index1, index2

class AttM(nn.Module):
    def __init__(self, length=5, hidden_dim=10, reduce=False):
        super(AttM, self).__init__()
        self.length1 = length
        self.length2 = length if not reduce else length - 1
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, team1, team2):
        assert team1.shape == team2.shape
        length1, length2 = self.length1, self.length2
        team1 = team1.view(-1, length1, length2, self.hidden_dim)
        team2 = team2.view(-1, length1, length2, self.hidden_dim)
        score = (self.W(team1) * team2).sum(dim=3)  # [b, 5, 5]
        score = F.softmax(score, dim=2)  # 固定 dim=2
        return score.view(-1, length1 * length2)  # [b, 25]

class FeatureInteraction(nn.Module):
    def __init__(self, n_player, player_dim, feature_dim=33, context_dim=1, 
                 intermediate_dim=16,  team_size=5, mlp_hidden_dim=50, dropout_rate=0.2,need_att=True):
        super(FeatureInteraction, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.team_size = team_size
        self.feature_dim = feature_dim 
        self.context_dim = context_dim
        self.index1, self.index2 = combine(team_size)

        # 選手嵌入層
        self.embedding = nn.Embedding(n_player, player_dim)

        # blade 和 chest 的輸入都統一：player_dim + feature_dim + context_dim
        input_dim = player_dim + feature_dim + context_dim
        self.attenM = AttM(length=team_size, hidden_dim=intermediate_dim, reduce=False)
        self.need_att = need_att
        self.blade_emb = nn.Sequential(
            nn.Linear(input_dim, 2*intermediate_dim),
            nn.SiLU(),
            nn.Linear(2*intermediate_dim,intermediate_dim)
        )
      
        self.chest_emb = nn.Sequential(
            nn.Linear(input_dim, 2 * intermediate_dim),
            nn.SiLU(),
            nn.Linear(2 * intermediate_dim, intermediate_dim)
        )

        self.mlp_pairwise_multiplication = nn.Sequential(# 兩兩配對相乘用
            nn.Linear(intermediate_dim, mlp_hidden_dim),  
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, 1, bias=True),
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, team_A_ids, team_B_ids, feats_A, feats_B, z_g_A, z_g_B):
        """
        team_A_ids: [batch_size, team_size]
        team_B_ids: [batch_size, team_size]
        feats_A:    [batch_size, team_size, feature_dim]
        feats_B:    [batch_size, team_size, feature_dim]
        z_g_A, z_g_B: [batch_size] (各自球隊的 context)
        """
        batch_size = feats_A.shape[0]

        # 選手嵌入向量
        team_A_emb = self.embedding(team_A_ids)  # [batch_size, team_size, player_dim]
        team_B_emb = self.embedding(team_B_ids)  # [batch_size, team_size, player_dim]

        # 將 z_g 擴展到 [batch_size, team_size, 1]
        z_g_A_exp = z_g_A.unsqueeze(-1).unsqueeze(-1).expand(-1, self.team_size, 1)  # [b, team_size, 1]
        z_g_B_exp = z_g_B.unsqueeze(-1).unsqueeze(-1).expand(-1, self.team_size, 1)  # [b, team_size, 1]

        # CONCAT： [team_emb, 33, context]
        A_concat = torch.cat([team_A_emb, feats_A, z_g_A_exp], dim=-1)  # [b, team_size, player_dim + 33 + 1]
        B_concat = torch.cat([team_B_emb, feats_B, z_g_B_exp], dim=-1)  # [b, team_size, player_dim + 33 + 1]
        a_blade = self.blade_emb(A_concat)  # [b, team_size, hidden_dim]
        b_chest = self.chest_emb(B_concat)  # [b, team_size, hidden_dim]

        # 配對交互：所有 (i, j) 兩兩相乘
        a_blade_pair = a_blade[:, self.index1, :]  # [b, team_size*team_size, hidden_dim]
        b_chest_pair = b_chest[:, self.index2, :]  # [b, team_size*team_size, hidden_dim]
        # 合併向量並計算分數
        interact = torch.mul(a_blade_pair, b_chest_pair)  # [b, team_size*team_size, hidden_dim]
        score = self.mlp_pairwise_multiplication(interact).squeeze(-1)  # [b, team_size*team_size]

        if self.need_att:
            normal = self.attenM(a_blade_pair, b_chest_pair)  # [b, team_size*team_size]
            score = score * normal  # 加權分數

        return score.sum(dim=1, keepdim=True)  # [b, 1]

class FIModel(nn.Module):
    def __init__(self, n_player, player_dim, feature_dim=33, context_dim=1,
                 intermediate_dim=16, team_size=5, mlp_hidden_dim=50, dropout_rate=0.2,
                 device=torch.device('cpu'), ema_tensor_path=None, game_id_mapping_path=None, need_att=True):
        super(FIModel, self).__init__()
        self.team_size = team_size
        self.device = device
        self.feature_dim = feature_dim
        self.context_dim = context_dim

        # 互動模組：現在只需要 player_dim、feature_dim、context_dim 等
        self.interaction = FeatureInteraction(
            n_player=n_player,
            player_dim=player_dim,
            feature_dim=feature_dim,
            context_dim=context_dim,
            intermediate_dim=intermediate_dim,
            team_size=team_size,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout_rate=dropout_rate,
            need_att= need_att
        )

        self.final_linear = nn.Linear(2, 1)

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
        
        # 將 game_ids 映射到張量索引
        game_indices = torch.tensor(
            [self.game_id_to_index.get(str(int(gid.item())), -1) for gid in game_ids], 
            dtype=torch.long, device=self.device
        )
        invalid_indices = game_indices == -1
        if invalid_indices.any():
            print(f"Warning: {invalid_indices.sum()} invalid game_ids")
            game_indices[invalid_indices] = 0  # 用 0 填充無效索引

        # 從 game_ema_tensor 提取特徵：[batch_size, 10, 全部特徵維度]
        features = self.game_ema_tensor[game_indices]  # [b, 10, ?]
        feats_A = features[:, :self.team_size, :self.feature_dim].clone()  # [b, 5, 33]
        feats_B = features[:, self.team_size:, :self.feature_dim].clone()  # [b, 5, 33]

        return feats_A, feats_B

    def forward(self, data):
        batch_size = data.shape[0]
        data = torch.LongTensor(data).to(self.device)
        
        # 主場(Ａ) context：z_g_off=1, z_g_def=0；客場(Ｂ)則互換
        z_g_off = torch.ones(batch_size, dtype=torch.float32).to(self.device)   # 主場進攻
        z_g_def = torch.zeros(batch_size, dtype=torch.float32).to(self.device)  # 主場防守
        z_g_off_opp = torch.zeros(batch_size, dtype=torch.float32).to(self.device)  # 客場進攻
        z_g_def_opp = torch.ones(batch_size, dtype=torch.float32).to(self.device)   # 客場防守
        
        # 分離 game_ids 和 player IDs
        game_ids = data[:, 0]
        player_data = data[:, 1:]
        
        team_A_ids = player_data[:, :self.team_size]  # [b, 5]
        team_B_ids = player_data[:, self.team_size:]  # [b, 5]
        
        # 載入完整EMA 特徵
        feats_A, feats_B = self.load_ema_features(player_data, game_ids)  # [b, 5, 33]
        
        # 主隊進攻 vs 客隊防守
        v_score = self.interaction(
            team_A_ids, team_B_ids, feats_A, feats_B, z_g_off, z_g_def
        )  # [b, 1]

        # 客隊進攻 vs 主隊防守
        v_opp_score = self.interaction(
            team_B_ids, team_A_ids, feats_B, feats_A, z_g_off_opp, z_g_def_opp
        )  # [b, 1]

        probs = torch.sigmoid(v_score - v_opp_score).squeeze(-1)  # [b]

        return probs, v_score, v_opp_score