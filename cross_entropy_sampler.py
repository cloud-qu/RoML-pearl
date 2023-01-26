
import numpy as np
from scipy import stats
import cross_entropy_method as cem


def get_cem_sampler(env_name, seed=None, alpha=0.05):
    get_nm = lambda tit: f'{tit}_{seed:d}' if seed is not None else tit

    if 'HalfCheetahVel' in env_name:
        return CEM_Beta(
            0.5, ref_alpha=alpha, batch_size=8*16,
            n_orig_per_batch=0.2, soft_update=0.5, title=get_nm('hc_vel'))
    elif 'HalfCheetahMass' in env_name:
        return LogBeta1D(
            0.5, ref_alpha=alpha, batch_size=8*16,
            n_orig_per_batch=0.2, soft_update=0.5, title=get_nm('hc_mass'))
    elif 'HalfCheetahBody' in env_name:
        return LogBeta(
            0.5*np.ones(3), ref_alpha=alpha, batch_size=8*16,
            n_orig_per_batch=0.2, soft_update=0.5, title=get_nm('hc_body'),
            titles=('mass', 'damping', 'head_size'))
    elif 'HumanoidVel' in env_name:
        return CEM_Beta(
            0.5, vmax=2.5, ref_alpha=alpha, batch_size=8*16,
            n_orig_per_batch=0.2, soft_update=0.5, title=get_nm('hum_vel'))
    elif 'HumanoidMass' in env_name:
        return LogBeta1D(
            0.5, ref_alpha=alpha, batch_size=8*16,
            n_orig_per_batch=0.2, soft_update=0.5, title=get_nm('hum_mass'))
    elif 'HumanoidBody' in env_name:
        return LogBeta(
            0.5 * np.ones(3), ref_alpha=alpha, batch_size=8*16,
            n_orig_per_batch=0.2, soft_update=0.5, title=get_nm('hum_body'),
            titles=('mass', 'damping', 'head_size'))
    elif env_name == 'KhazadDum-v0':
        return CemExp(
            0.1, ref_alpha=alpha, batch_size=8*16, min_batch_update=0.05,
            n_orig_per_batch=0.2, soft_update=0.5, title=get_nm('kd'))
    raise ValueError(env_name)


class CEM_Beta(cem.CEM):
    '''CEM for 1D Beta distribution.'''

    def __init__(self, *args, vmax=7.0, eps=0.02, **kwargs):
        super(CEM_Beta, self).__init__(*args, **kwargs)
        self.default_dist_titles = 'beta_mean'
        self.default_samp_titles = 'sample'
        self.vmax = vmax
        self.eps = eps

    def do_sample(self, phi):
        return self.vmax * np.random.beta(2*phi, 2-2*phi)

    def pdf(self, x, phi):
        return stats.beta.pdf(np.clip(x/self.vmax,self.eps,1-self.eps),
                              2*phi, 2-2*phi)

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples) / self.vmax
        return np.clip(np.mean(w*s)/np.mean(w), self.eps, 1-self.eps)

class LogBeta1D(cem.CEM):
    '''x ~ 2**Beta'''

    def __init__(self, *args, log_range=(-1, 1), eps=0.02, **kwargs):
        super(LogBeta1D, self).__init__(*args, **kwargs)
        self.default_dist_titles = 'beta_mean'
        self.default_samp_titles = 'sample'
        self.log_range = log_range
        self.eps = eps

    def do_sample(self, phi):
        x = np.random.beta(2*phi, 2-2*phi)
        x = self.log_range[0] + (self.log_range[1]-self.log_range[0]) * x
        return np.power(2, x)

    def pdf(self, x, phi):
        x = np.log2(x)
        x = (x - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return stats.beta.pdf(np.clip(x, self.eps, 1-self.eps),
                              2*phi, 2-2*phi)

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        s = (np.log2(s) - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return np.clip(np.mean(w*s)/np.mean(w), self.eps, 1-self.eps)

class LogBeta(cem.CEM):
    '''x ~ 2**Beta, multi-dimensional'''

    def __init__(self, *args, log_range=(-1, 1), eps=0.02, titles=None, **kwargs):
        super(LogBeta, self).__init__(*args, **kwargs)
        if titles is not None:
            self.default_dist_titles = [f'{t}_mean' for t in titles]
            self.default_samp_titles = [t for t in titles]
        self.log_range = log_range
        self.eps = eps

    def do_sample(self, phi):
        x = np.random.beta(2*phi, 2-2*phi)
        x = self.log_range[0] + (self.log_range[1]-self.log_range[0]) * x
        return np.power(2, x)

    def pdf(self, x, phi):
        x = np.log2(x)
        x = (x - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return stats.beta.pdf(np.clip(x, self.eps, 1-self.eps),
                              2*phi, 2-2*phi).prod()

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        wmean = np.mean(w)
        w = np.repeat(w[:, np.newaxis], s.shape[1], axis=1)
        s = (np.log2(s) - self.log_range[0]) / (self.log_range[1]-self.log_range[0])
        return np.clip(np.mean(w*s, axis=0)/wmean, self.eps, 1-self.eps)

class CemExp(cem.CEM):
    def __init__(self, *args, **kwargs):
        super(CemExp, self).__init__(*args, **kwargs)
        self.default_dist_titles = 'exp_mean'
        self.default_samp_titles = 'sample'

    def do_sample(self, phi):
        return np.random.exponential(phi)

    def pdf(self, x, phi):
        return stats.expon.pdf(x, 0, phi)

    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
        return np.mean(w*s)/np.mean(w)
