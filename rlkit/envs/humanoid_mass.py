import numpy as np
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv

from . import register_env

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


@register_env('humanoid-mass')
class HumanoidMassEnv(HumanoidEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._idx = 0
        if not self._task:
            self._task = self.tasks[0]
        super(HumanoidMassEnv, self).__init__()

        self.original_mass = self.model.body_mass.copy()
        self.original_size = self.model.geom_size.copy()

        self._curr_steps = n_tasks * [0]
        self._curr_return = n_tasks * [0]

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim))
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)

        qpos = self.sim.data.qpos
        alive = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        alive_bonus = 5.0 if alive else 0.0
        done = False

        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        self._curr_steps[self._idx] += 1
        self._curr_return[self._idx] += reward
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_task_return(self, idx=None):
        if idx is None:
            idx = self._idx
        return self._curr_return[idx]

    def sample_tasks(self, num_tasks):
        #np.random.seed(1337)
        factors =  2 ** np.random.uniform(-1, 1, size=(num_tasks,))
        tasks = [{'mass_factor': fac} for fac in factors]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx, task=None, resample_task=False):
        self._idx = idx
        if task is not None:
            if not isinstance(task, dict):
                task = dict(mass_factor=task)
            self.tasks[idx] = task
        elif resample_task:
            self.tasks[idx] = self.sample_tasks(1)[0]
        self._task = self.tasks[idx]
        self._curr_steps[self._idx] = 0
        self._curr_return[self._idx] = 0
        self.set_task()
        self.reset()

    def set_task(self):
        task = self._task['mass_factor']
        for i in range(len(self.model.body_mass)):
            self.model.body_mass[i] = task * self.original_mass[i]
        self.model.geom_size[1:, 0] = task * self.original_size[1:, 0]
        return task
