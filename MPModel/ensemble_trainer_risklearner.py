import torch
import torch.nn.functional as F
import pdb
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
import wandb
import numpy as np
from .diverse_trainer_risklearner import DiverseRiskLearnerTrainer
class EnsembleRiskLearnerTrainer(DiverseRiskLearnerTrainer):
    """
    Class to handle training of RiskLearner for functions.
    """

    def __init__(self, device, risklearner, optimizer, real_batch_size, num_ensemble=10, output_type="deterministic", kl_weight=0.005, num_subset_candidates=200000):

        self.device = device
        self.risklearner = risklearner
        self.optimizer = optimizer
        self.real_batch_size = real_batch_size
        self.num_ensemble = num_ensemble
        self.num_subset_candidates = num_subset_candidates

        # ++++++Prediction distribution p(l|tau)++++++++++++++++++++++++++++
        self.output_type = output_type
        self.kl_weight = kl_weight

        # ++++++Acquisition functions++++++++++++++++++++++++++++
        self.acquisition_type = "lower_confidence_bound"

    def train(self, Risk_X, Risk_Y):
        Risk_X, Risk_Y = Risk_X.unsqueeze(0), Risk_Y.unsqueeze(0).unsqueeze(-1)
        # shape: 1, batch_size, dim

        self.optimizer.zero_grad()
        p_y_pred = self.risklearner(Risk_X, self.output_type)#num_ensemble, batch_size, 1
        y_label = Risk_Y.repeat(self.num_ensemble, 1, 1)#num_ensemble, batch_size, 1

        loss = F.mse_loss(p_y_pred, y_label, reduction="mean")
        loss.backward()
        self.optimizer.step()

        return loss
    
    def acquisition_function(self, Risk_X_candidate, gamma_0=1.0, gamma_1=1.0, gamma_2=0.0, pure_acquisition=False):

        Risk_X_candidate = Risk_X_candidate.to(self.device)
        x = Risk_X_candidate.unsqueeze(0)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        # Shape: 1 * bs * dim

        if self.acquisition_type == "lower_confidence_bound":

            p_y_pred = self.risklearner(x, self.output_type)
            # Shape: num_ensembles * batch_size * 1

            output_mu = torch.mean(p_y_pred, dim=0)
            output_sigma = torch.std(p_y_pred, dim=0)
            acquisition_score = gamma_0 * output_mu + gamma_1 * output_sigma
            
            if pure_acquisition:
                return acquisition_score, output_mu, output_sigma
            
            best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score = self.diversified_score(x.squeeze(0), acquisition_score, gamma_2)

        return best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score