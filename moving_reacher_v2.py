import numpy as np
import mujoco
import gymnasium as gym
import pygame
from gymnasium import utils
from gymnasium.spaces import Box
from mujoco_env_custom import MujocoSim #Mujoco physics simulation that serves as the environment that this agent acts on

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}

class MovingReacherV2(MujocoSim, utils.EzPickle, gym.Env):
    # Parameters for object
    metadata = {
        "render_modes":[
            "human", 
            "rgb_array",
            "depth_array"
        ],
        "render_fps": 120,
    }

    #arm_length: an integer representing the total length of reacher's arm. Each bar then has length = arm_length/2
    #render_mode: 'human' means we render the mujoco model for visual representation. [MORE DESCRIPTION COMING SOON]
    #
    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        observation_space = Box(low=(-np.inf), high=(np.inf), shape = (14,), dtype=np.float64)
        MujocoSim.__init__(
            self,
            "reacher_model.xml",
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, act):
        dist_vec = self.get_body_com("tip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(dist_vec)
        reward_ctrl = -np.square(act).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(act, self.frame_skip)
        if self.render_mode == "human":
            self.render()
        
        obs = self._get_obs()

        return (
            obs,
            reward,
            False,
            False,
            dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        )

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qpos.flat[2:],
                self.data.qvel.flat[:2],
                self.get_body_com("tip") - self.get_body_com("target"),
                self.get_body_com("base_slider"),
            ]
        )
    

##Testing
            
env_test = MovingReacherV2(render_mode="human")
assert env_test is not None, "Failed to create modified env"

initial_observation, _ = env_test.reset()
assert initial_observation is not None, "Reset method should return an initial observation"

action = env_test.action_space.sample()  # Generate a random action
observation, reward, terminated, truncated, info = env_test.step(action)
assert observation is not None, "Step method should return an observation"
assert isinstance(reward, float), "Step method should return a reward as float"
assert isinstance(terminated, bool), "Step method should return a 'done' flag as bool"
assert isinstance(truncated, bool), "Step method should return a 'done' flag as bool"
assert isinstance(info, dict), "Step method should return 'info' as a dictionary"

try:
    env_test.render()
except Exception as e:
    assert False, f"Rendering failed with an exception: {e}"

for episode in range(10):  # Test for 10 episodes
    observation = env_test.reset()
    done = False
    while not done:
        action = env_test.action_space.sample()
        observation, reward, terminated, truncated, info = env_test.step(action)
        env_test.render()
env_test.close()