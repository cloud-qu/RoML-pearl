"""
Launcher for experiments with PEARL

"""
import logging
import os
import pathlib
import numpy as np
import click
import json
import torch
import wandb

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config


def experiment(variant):
    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params'][
        'use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
            )
    qf1 = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            )
    qf2 = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            )
    vf = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size=obs_dim + latent_dim,
            output_size=1,
            )
    policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            )
    agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **variant['algo_params']
            )
    n_train = variant['n_train_tasks']
    n_valid = variant['n_eval_tasks']
    n_test = variant['n_test_tasks']
    variant['algo_params']['gpu_id'] = variant['util_params']['gpu_id']
    algorithm = PEARLSoftActorCritic(
            env=env,
            train_tasks=list(tasks[:n_train]),
            eval_tasks=list(tasks[n_train:n_train + n_valid]),
            test_tasks=list(tasks[-n_test:]),
            nets=[agent, qf1, qf2, vf],
            latent_dim=latent_dim,
            **variant['algo_params']
            )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        ptu.device = torch.device('cuda:{}'.format(variant['util_params']['gpu_id']))
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id,
                                      base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


@click.command()
@click.option('--config', default=None)
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
@click.option('--use_wandb', default=True)
@click.option('--n_test_tasks', default=None)
@click.option('--num_iterations', default=None)
@click.option('--seed', default=None)
@click.option('--use_cem', default=False)
@click.option('--framework', default=0, type=int)#0:pearl, 1:roml, 2:mpts
@click.option('--gamma_2', default=0, type=float)
@click.option('--diverse_sample_ratio', default=0, type=float)
@click.option('--posterior_sampling', default=1, type=int)
@click.option('--diversity_type', default='rs', type=str)
def main(config, gpu, docker, debug, use_wandb, n_test_tasks, num_iterations, seed, use_cem, framework, gamma_2, diverse_sample_ratio, diversity_type, posterior_sampling):
    if framework != 1:
        use_cem = False
    variant = default_config
    if n_test_tasks is not None:
        variant['n_test_tasks'] = int(n_test_tasks)
    if num_iterations is not None:
        variant['algo_params']['num_iterations'] = int(num_iterations)
    if use_cem is not None:
        variant['algo_params']['use_cem'] = bool(use_cem)
    if seed is not None:
        variant['algo_params']['seed'] = int(seed)
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['algo_params']['framework'] = framework
    variant['algo_params']['gamma_2'] = gamma_2
    variant['algo_params']['diverse_sample_ratio'] = diverse_sample_ratio
    variant['algo_params']['diversity_type'] = diversity_type if diversity_type != 'none' else None
    variant['algo_params']['posterior_sampling'] = posterior_sampling
    if use_wandb:
        ngc_run = os.path.isdir('/ws')
        if ngc_run:
            ngc_dir = '/result/wandb/'  # args.ngc_path
            os.makedirs(ngc_dir, exist_ok=True)
            logging.info('NGC run detected. Setting path to workspace: {}'.format(ngc_dir))
            wandb.init(project="roml", sync_tensorboard=True, config=variant, dir=ngc_dir)
        else:
            wandb.init(project="roml", sync_tensorboard=True, config=variant)
    experiment(variant)


if __name__ == "__main__":
    main()
