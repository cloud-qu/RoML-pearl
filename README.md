# Robust Meta Reinforcement Learning (RoML) with PEARL

RoML (Robust Meta Reinforcement Learning) is a meta-algorithm that takes a meta-learning baseline algorithm, and generates a robust version of it.

This repo implements RoML on top of the [official implementation](https://github.com/katerakelly/oyster) of [PEARL](https://arxiv.org/abs/1903.08254) algorithm for meta reinforcement learning.
To implement RoML we changed the tasks sampling procedure, by adding the file `cross_entropy_sampler.py` and using it in `rlkit/core/rl_algorithm.py` (search for "cem" in `rl_algorithm.py` to see the modifications).

See [here](https://github.com/ido90/RobustMetaRL) more details about what is RoML and how to use it in general.

### How to run

The API is identical to the original repo of PEARL, with the additional flag `use_cem`, which switches between RoML and the PEARL baseline.
To reproduce the experiments in our paper, run:

```python launch_experiment.py --config configs/ENV.json --seed SEED --use_cem IS_CEM```

where SEED is an integer, IS_CEM is either 0 or 1, and ENV is the desired environment: `cheetah-vel`, `cheetah-mass` or `cheetah-body`.
The results can be processed as shown in the notebook `RoML-PEARL-Mujoco.ipynb`.
