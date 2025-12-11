import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

class BNN_BT(nn.Module):
    def __init__(self, n_player, prior_mu=0.0, prior_sigma=1.0):
        super(BNN_BT, self).__init__()
        self.n_player = n_player
        self.mu = nn.Parameter(torch.ones(n_player) * prior_mu)
        self.rho = nn.Parameter(torch.ones(n_player) * torch.log(torch.tensor(2.71828 - 1)))
        self.beta = prior_sigma / 2
        self.beta2 = self.beta ** 2
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def sample_skills(self, num_samples=1, device='cuda:0'):
        sigma = torch.log1p(torch.exp(self.rho))  # [n_player]
        eps = torch.randn(num_samples, self.n_player, device=device)  # [num_samples, n_player]
        s_i = self.mu.unsqueeze(0) + eps * sigma.unsqueeze(0)  # [num_samples, n_player]
        return s_i

    def forward(self, team, num_samples=1):
        sampled_skills = self.sample_skills(num_samples, team.device)  # [num_samples, n_player]
        team_skills = sampled_skills[:, team]  # [num_samples, batch_size, team_size]
        team_performances = team_skills.sum(dim=-1)  # [num_samples, batch_size]
        return team_performances

    def kl_divergence(self):
        sigma = torch.log1p(torch.exp(self.rho))
        sigma = torch.clamp(sigma, min=1e-6)
        kl = 0.5 * torch.sum(
            (sigma / self.prior_sigma) ** 2 +
            (self.mu - self.prior_mu) ** 2 / (self.prior_sigma ** 2) -
            1 - 2 * torch.log(sigma / self.prior_sigma)
        )
        return kl

class NAC_BBB(nn.Module):
    def __init__(self, n_player, team_size=5, device=torch.device('cuda:0'), prior_mu=0.0, prior_sigma=1.0):
        super(NAC_BBB, self).__init__()
        self.n_player = n_player
        self.team_size = team_size
        self.device = device
        self.BT = BNN_BT(n_player, prior_mu, prior_sigma).to(device)

    def forward(self, data, num_samples=1):
        if not isinstance(data, torch.Tensor):
            data = torch.LongTensor(data).to(self.device)
        elif data.device != self.device:
            data = data.to(self.device)
        
        team_A = data[:, 1:1+self.team_size]  # [batch_size, team_size]
        team_B = data[:, 1+self.team_size:]  # [batch_size, team_size]
        sampled_skills = self.BT.sample_skills(num_samples, self.device)  # [num_samples, n_player]
        team_A_skills = sampled_skills[:, team_A]  # [num_samples, batch_size, team_size]
        team_B_skills = sampled_skills[:, team_B]  # [num_samples, batch_size, team_size]
        t_A = team_A_skills.sum(dim=-1)  # [num_samples, batch_size]
        t_B = team_B_skills.sum(dim=-1)  # [num_samples, batch_size]
        
        diff = t_A - t_B  # [num_samples, batch_size]
        sigma = torch.log1p(torch.exp(self.BT.rho))  # [n_player]
        sigma_A = (sigma[team_A]**2).sum(dim=-1)  # [batch_size]
        sigma_B = (sigma[team_B]**2).sum(dim=-1)  # [batch_size]
        variance = sigma_A + sigma_B + self.BT.beta2 * (self.team_size + self.team_size)  # [batch_size]
        variance = torch.clamp(variance, min=1e-6)
        variance = variance.unsqueeze(0).expand(num_samples, -1)  # [num_samples, batch_size]
        z = diff / torch.sqrt(variance)  # [num_samples, batch_size]
        prob = 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=z.device))))  # [num_samples, batch_size]
        return prob, z

    def kl_divergence(self):
        return self.BT.kl_divergence()

    def get_top_players(self, index_to_player_id, top_k=10):
        scores = self.BT.mu.detach().cpu().numpy()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_players = [(index_to_player_id[idx], float(scores[idx])) for idx in top_indices]
        return top_players