import torch
import torch.nn.functional as F
import pdb
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import wandb
import itertools
import numpy as np
class DiverseRiskLearnerTrainer():
    """
    Class to handle training of RiskLearner for functions.
    """

    def __init__(self, device, risklearner, optimizer, real_batch_size, output_type="deterministic", kl_weight=0.005, num_subset_candidates=200000):

        self.device = device
        self.risklearner = risklearner
        self.optimizer = optimizer
        self.real_batch_size = real_batch_size
        self.num_subset_candidates = num_subset_candidates

        # ++++++Prediction distribution p(l|tau)++++++++++++++++++++++++++++
        self.output_type = output_type
        self.kl_weight = kl_weight

        # ++++++initialize the p(z_0)++++++++++++++++++++++++++++
        r_dim = self.risklearner.r_dim
        prior_init_mu = torch.zeros([1, r_dim]).to(self.device)
        prior_init_sigma = torch.ones([1, r_dim]).to(self.device)
        self.z_prior = Normal(prior_init_mu, prior_init_sigma)

        # ++++++Acquisition functions++++++++++++++++++++++++++++
        self.acquisition_type = "lower_confidence_bound"
        self.num_samples = 50

    def train(self, Risk_X, Risk_Y):
        Risk_X, Risk_Y = Risk_X.unsqueeze(0), Risk_Y.unsqueeze(0).unsqueeze(-1)
        # shape: batch_size, num_points, dim

        self.optimizer.zero_grad()
        p_y_pred, z_variational_posterior = self.risklearner(Risk_X, Risk_Y, self.output_type)
        z_prior = self.z_prior

        loss, recon_loss, kl_loss = self._loss(p_y_pred, Risk_Y, z_variational_posterior, z_prior)
        loss.backward()
        self.optimizer.step()

        # updated z_prior
        self.z_prior = Normal(z_variational_posterior.loc.detach(), z_variational_posterior.scale.detach())

        return loss, recon_loss, kl_loss
    def _loss(self, p_y_pred, y_target, posterior, prior):

        negative_log_likelihood = F.mse_loss(p_y_pred, y_target, reduction="mean")
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(posterior, prior).mean(dim=0).sum()

        return negative_log_likelihood + kl * self.kl_weight, negative_log_likelihood, kl
    
    # def diversified_score(self, Risk_X_candidate, acquisition_score, gamma_2):
    #     x = Risk_X_candidate.squeeze(0)  # bs, dim
    #     num_candidates = len(x)
    #     num_samples = self.num_subset_candidates

    #     indices = np.array([np.random.choice(num_candidates, self.real_batch_size, replace=False) for _ in range(num_samples)])
    #     sampled_combinations = indices #torch.tensor(indices).view(num_samples, self.real_batch_size)

    #     # 计算多样性得分
    #     x_expanded = x[sampled_combinations]  # (num_samples, real_batch_size, dim)
    #     x_diff = x_expanded.unsqueeze(2) - x_expanded.unsqueeze(1)  # (num_samples, real_batch_size, real_batch_size, dim)
    #     local_diverse_score = torch.norm(x_diff, dim=-1).sum(dim=(1, 2)) * gamma_2  # (num_samples,)
        
    #     # 计算获取得分
    #     local_acquisition_score = acquisition_score[sampled_combinations].sum(dim=1).squeeze()  # (num_samples,)
        
    #     # 计算总得分
    #     combine_subset_acquisition_score = local_acquisition_score + local_diverse_score  # (num_samples,)
        
    #     # 找到最佳组合
    #     best_batch_id = sampled_combinations[torch.argmax(combine_subset_acquisition_score)]  # (real_batch_size,)
        
    #     return best_batch_id, combine_subset_acquisition_score[torch.argmax(combine_subset_acquisition_score)].item(), local_diverse_score[torch.argmax(combine_subset_acquisition_score)].item(), local_acquisition_score[torch.argmax(combine_subset_acquisition_score)].item()

    def diversified_score(self, Risk_X_candidate, acquisition_score, gamma_2, real_batch_size=None):
        x = Risk_X_candidate.squeeze(0)  # bs, dim
        x = x.cpu().numpy()
        acquisition_score = acquisition_score.cpu().detach().numpy()
        num_candidates = len(x)
        num_samples = self.num_subset_candidates
        if real_batch_size is None:
            real_batch_size = self.real_batch_size

        # 使用 numpy 进行随机采样
        indices = np.array([np.random.choice(num_candidates, real_batch_size, replace=False) for _ in range(num_samples)])
        
        # 计算多样性得分
        x_expanded = x[indices]  # (num_samples, real_batch_size, dim)
        x_diff = x_expanded[:, :, np.newaxis, :] - x_expanded[:, np.newaxis, :, :]  # (num_samples, real_batch_size, real_batch_size, dim)
        local_diverse_score = np.linalg.norm(x_diff, axis=-1).sum(axis=(1, 2)) / ((real_batch_size) * (real_batch_size - 1)) * gamma_2  # (num_samples,)
        
        # 计算获取得分
        local_acquisition_score = acquisition_score[indices].sum(axis=1).squeeze()  # (num_samples,)
        
        # 计算总得分
        combine_subset_acquisition_score = local_acquisition_score + local_diverse_score  # (num_samples,)
        
        # 找到最佳组合
        best_idx = np.argmax(combine_subset_acquisition_score)
        best_batch_id = indices[best_idx]  # (real_batch_size,)
        
        return best_batch_id, combine_subset_acquisition_score[best_idx], local_diverse_score[best_idx], local_acquisition_score[best_idx]

    def acquisition_function(self, Risk_X_candidate, gamma_0=1.0, gamma_1=1.0, gamma_2=0.0, pure_acquisition=False, real_batch_size=None):

        Risk_X_candidate = Risk_X_candidate.to(self.device)
        x = Risk_X_candidate.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        # Shape: 1 * 100 * 2

        if self.acquisition_type == "lower_confidence_bound":
            z_sample = self.z_prior.rsample([self.num_samples])
            # Shape: num_samples * 1 * 10

            p_y_pred = self.risklearner.xz_to_y(x, z_sample, self.output_type)
            # Shape: num_samples * batch_size * 1

            output_mu = torch.mean(p_y_pred, dim=0)#bs, 1
            output_sigma = torch.std(p_y_pred, dim=0)#bs, 1
            acquisition_score = gamma_0 * output_mu + gamma_1 * output_sigma
            if pure_acquisition:
                return acquisition_score, output_mu, output_sigma
            
            
            best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score = self.diversified_score(x, acquisition_score, gamma_2, real_batch_size=real_batch_size)

        return best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score