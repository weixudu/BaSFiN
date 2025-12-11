import os
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from BaS import NAC_BBB
from bc_fim2 import FIModel
from co_fim2 import NAC_ANFM

logger = logging.getLogger(__name__)


class NAC(nn.Module):
    def __init__(
        self,
        n_hero: int,
        *,
        team_size: int = 5,
        anfm_hidden_dim: int = 32,
        intermediate_dim: int = 64,
        prob_dim: int = 16,
        bc_player_dim: int = 16,
        anfm_player_dim: int = 16,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        kl_weight: float = 0.01,
        dropout: float = 0.2,
        need_att: bool = False,
        num_samples: int = 100,
        anfm_drop: float = 0.3,
        fim_drop: float = 0.3,
        anfm_mlp: int = 16,
        fim_mlp: int = 16,
        bc_need_att: bool = True,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ema_tensor_path: Optional[str] = None,
        game_id_mapping_path: Optional[str] = None,
        model_save_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.team_size = team_size
        self.num_samples = num_samples
        self.kl_weight = kl_weight

        # ---------------- 子模組 ---------------- #
        self.nac_bbb = NAC_BBB(n_hero, team_size, device, prior_mu, prior_sigma)
        self.fimodel = FIModel(
            n_player=n_hero,
            player_dim=bc_player_dim,
            intermediate_dim=intermediate_dim,
            team_size=team_size,
            mlp_hidden_dim=fim_mlp,
            dropout_rate=fim_drop,
            device=device,
            ema_tensor_path=ema_tensor_path,
            game_id_mapping_path=game_id_mapping_path,
            need_att=bc_need_att,
        )
        self.nac_anfm = NAC_ANFM(
            n_player=n_hero,
            player_dim=anfm_player_dim,
            team_size=team_size,
            hidden_dim=anfm_hidden_dim,
            need_att=need_att,
            mlp_hidden_dim=anfm_mlp,
            dropout_rate=anfm_drop,
            device=device,
            ema_tensor_path=ema_tensor_path,
            game_id_mapping_path=game_id_mapping_path,
        )

        # --------- (可選) 載入預訓練權重 --------- #
        if model_save_dir:
            for sub, fn in zip(
                (self.nac_bbb, self.fimodel, self.nac_anfm),
                ("nac_bbb.pth", "fimodel.pth", "anfm.pth")
            ):
                path = os.path.join(model_save_dir, fn)
                if os.path.isfile(path):
                    try:
                        sub.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                    except Exception as e:
                        logger.warning(f"Load {fn} failed: {e}")

        # ---------------- 最終 MLP ---------------- #
        self.final_mlp = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )


    def _initialize_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
                
        self.apply(self._initialize_weights)
        self.to(self.device)

    # ------------------------------------------------------------------ #
    #                               Forward                              #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        data: torch.Tensor,
        training: bool = True,
        *,
        return_module_contrib: bool = False,
        return_feature_contrib: bool = False,
        return_param_contrib: bool = False,
    ) -> Tuple[
        torch.Tensor,                # probs [S,B]
        Optional[torch.Tensor],      # module_grad [B,3]
        Optional[torch.Tensor],      # feature_grad [B,5]
        Optional[torch.Tensor],      # param_grad   [3]
    ]:
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.long, device=self.device)
        elif data.device != self.device:
            data = data.to(self.device)

        S = self.num_samples if training else 1
        B = data.size(0)

        # -------- 子模組模式切換 -------- #
        self.nac_bbb.train(training)
        self.fimodel.train(training)
        self.nac_anfm.train(training)

        # -------- 子模組輸出 -------- #
        _, z = self.nac_bbb(data, num_samples=S)         # [S,B]
        _, comp_a, comp_b = self.fimodel(data)           # [B,1]
        _, coop_a, coop_b = self.nac_anfm(data)          # [B,1]

        # -------- reshape → [S,B,1] -------- #
        z      = z.unsqueeze(-1)
        comp_a = comp_a.unsqueeze(0).expand(S, -1, -1)
        comp_b = comp_b.unsqueeze(0).expand(S, -1, -1)
        coop_a = coop_a.unsqueeze(0).expand(S, -1, -1)
        coop_b = coop_b.unsqueeze(0).expand(S, -1, -1)

        # -------- 只用原始五項 -------- #
        feature_tensor = torch.cat([z, comp_a, comp_b, coop_a, coop_b], dim=-1)  # [S,B,5]

        # -------- 梯度追蹤判斷 -------- #
        need_grad = return_module_contrib or return_feature_contrib or return_param_contrib
        if need_grad:
            if not feature_tensor.requires_grad:
                feature_tensor = feature_tensor.detach().requires_grad_(True)
            else:
                feature_tensor.retain_grad()

        # -------- 推論 -------- #
        logits = feature_tensor.view(S * B, 5)
        probs  = self.final_mlp(logits).squeeze(-1).view(S, B)

        # -------- 梯度訊號計算 -------- #
        module_grad = feature_grad = param_grad = None
        if need_grad:
            mean_prob = probs.mean()
            self.zero_grad(set_to_none=True)
            mean_prob.backward()

            if return_feature_contrib or return_module_contrib:
                feature_grad = feature_tensor.grad.detach().mean(dim=0)  # [B,5]

            if return_module_contrib:
                z_idx     = [0]
                comp_idx  = [1, 2]
                coop_idx  = [3, 4]
                z_grad    = feature_grad[:, z_idx].sum(dim=-1, keepdim=True)
                comp_grad = feature_grad[:, comp_idx].sum(dim=-1, keepdim=True)
                coop_grad = feature_grad[:, coop_idx].sum(dim=-1, keepdim=True)
                module_grad = torch.cat([z_grad, comp_grad, coop_grad], dim=-1)  # [B,3]

            if return_param_contrib:
                grads = []
                for sub in (self.nac_bbb, self.fimodel, self.nac_anfm):
                    tot = 0.0
                    for p in sub.parameters():
                        if p.grad is not None:
                            tot += p.grad.detach().abs().sum()
                    grads.append(tot)
                param_grad = torch.tensor(grads, device=self.device)  # [3]

        return probs, module_grad, feature_grad, param_grad

    # ------------------------------------------------------------------ #
    #                            Loss & Utils                            #
    # ------------------------------------------------------------------ #
    def kl_divergence(self) -> torch.Tensor:
        return self.nac_bbb.kl_divergence()

    def elbo_loss(self, probs: torch.Tensor, y, num_samples: int) -> torch.Tensor:
        y_tensor = torch.as_tensor(y, dtype=torch.float, device=self.device)
        y_rep = y_tensor.unsqueeze(0).repeat(num_samples, 1)
        nll = F.binary_cross_entropy(probs, y_rep, reduction="sum") / num_samples
        return nll + self.kl_weight * self.kl_divergence()

