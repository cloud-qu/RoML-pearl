import torch
import torch.nn as nn
import pdb
from torch.distributions import Normal
from MPModel.backbone_risklearner import Ensemble_Predictor

# Define RiskLearner for the current task distribution and the current backbone parameters...
class BaseRiskLearner(nn.Module):
    """
    Implements risklearner for functions of arbitrary dimensions.
    x_dim : int Dimension of x values.
    y_dim : int Dimension of y values.
    r_dim : int Dimension of output representation r.
    z_dim : int Dimension of latent variable z.
    h_dim : int Dimension of hidden layer in encoder and decoder.
    """

    def __init__(self, x_dim, y_dim, h_dim):
        super(BaseRiskLearner, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim

        # Initialize networks
        self.x_to_y = Ensemble_Predictor(x_dim, h_dim, y_dim)

    def forward(self, x, output_type):
        """
        returns a distribution over target points y_target. We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        #1, num, x_dim

        if self.training:
            p_y_pred = self.x_to_y(x, output_type) #1, num, y_dim
            return p_y_pred


class Ensemble_RiskLearner(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, num_ensemble=10):
        super(Ensemble_RiskLearner, self).__init__()
        self.num_ensemble = num_ensemble
        self.risklearners = nn.ModuleList([BaseRiskLearner(x_dim, y_dim, h_dim) for _ in range(num_ensemble)])

    def forward(self, x, output_type):
        p_y_preds = []
        for risklearner in self.risklearners:
            p_y_pred = risklearner(x, output_type)
            p_y_preds.append(p_y_pred)
        p_y_preds = torch.cat(p_y_preds, dim=0)
        return p_y_preds
    
