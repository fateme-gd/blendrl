from typing import Sequence
import torch
from nudge.env_vectorized import VectorizedNudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np
import torch as th
from ocatari.ram.seaquest import MAX_ESSENTIAL_OBJECTS
import gymnasium
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from utils import load_cleanrl_envs


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


# def make_env(env):
#     # def thunk():
#         # if capture_video and idx == 0:
#             # env = gym.make(env_id, render_mode="rgb_array")
#             # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         # else:
#             # env = gym.make(env_id)
#     env = NoopResetEnv(env, noop_max=30)
#     env = MaxAndSkipEnv(env, skip=4)
#     env = EpisodicLifeEnv(env)
#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)
#     env = ClipRewardEnv(env)
#     env = gym.wrappers.ResizeObservation(env, (84, 84))
#     env = gym.wrappers.GrayScaleObservation(env)
#     env = gym.wrappers.FrameStack(env, 4)
#     env = gym.wrappers.AutoResetWrapper(env)
#     env = gym.wrappers.RecordEpisodeStatistics(env)
#     return env

def make_env(env):
    # def thunk():
        # if capture_video and idx == 0:
            # env = gym.make(env_id, render_mode="rgb_array")
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
            # env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AutoResetWrapper(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "seaquest"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'up': 2,
        'right': 3,
        'left': 4,
        'down': 5,
    }
    pred_names: Sequence

    def __init__(self, mode: str, n_envs: int, render_mode="rgb_array", render_oc_overlay=False, seed=None):
        super().__init__(mode)
        #self.raw_env = gymnasium.make("SeaquestNoFrameskip-v4")
        # self.raw_env = gymnasium.make("Seaquest-v4")
        # self.raw_env = load_cleanrl_envs("Seaquest-v4")
        # self.raw_env = load_cleanrl_envs("SeaquestDeterministic-v4")
        # self.raw_env = env = make_atari_env('SeaquestNoFrameskip-v4', n_envs=1, seed=seed)
        # self.raw_env = VecFrameStack(env, n_stack=4)

        # self.env = OCAtari(env_name="SeaquestNoFrameskip-v4", mode="ram",
        # self.env = OCAtari(env_name="Seaquest-ramDeterministic-v4", mode="ram",
        # self.env = OCAtari(env_name="Seaquest", mode="ram",
        # self.env = OCAtari(env_name="Seaquest-v4", mode="ram",
        #                    render_mode=render_mode, render_oc_overlay=render_oc_overlay)
        self.n_envs = n_envs
        self.envs = [OCAtari(env_name="Seaquest-v4", mode="ram", obs_mode="ori",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay) for i in range(n_envs)]
        # apply wrapper to _env in OCAtari
        for i in range(n_envs):
            self.envs[i]._env = make_env(self.envs[i]._env)
        
        # for learning script from cleanrl
        # self.env._env = make_env(self.env._env)
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 43
        self.n_features = 4  # visible, x-pos, y-pos, right-facing
        self.seed = seed

        # Compute index offsets. Needed to deal with multiple same-category objects
        self.obj_offsets = {}
        offset = 0
        for (obj, max_count) in MAX_ESSENTIAL_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_ESSENTIAL_OBJECTS.keys())

    def reset(self):
        logic_states = []
        neural_states = []
        seed_i = self.seed
        for env in self.envs:
            obs, _ = env.reset(seed=seed_i)
            # lazy frame to tensor
            obs = torch.tensor(obs).float()
            state = env.objects
            raw_state = obs #self.env.dqn_obs
            logic_state, neural_state =  self.extract_logic_state(state), self.extract_neural_state(raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            seed_i += 1
        return torch.stack(logic_states), torch.stack(neural_states)
        # raw_state, _ = self.raw_env.reset(seed=self.seed)
        # raw_state = raw_state.unsqueeze(0)
        # if len(logic_state.shape) == 2:
        # logic_state = logic_state.unsqueeze(0)
        # return logic_state, neural_state
        # return  self.convert_state(state, raw_state)

    def step(self, actions, is_mapped: bool = False):
        # if not is_mapped:
        #     action = self.map_action(action)
        # step RAM env
        # obs, reward, done, _, _ = self.env.step(action)
        # action = array([2]) or action = torch.tensor(2)
        # try:
        #     assert action.shape[0] == 1, "invalid only 1 action for env.step"
        #     action = action[0]
        # except IndexError:
        #     action = action
        assert len(actions) == self.n_envs, "Invalid number of actions: n_actions is {} and n_envs is {}".format(len(actions), self.n_envs)
        observations = []
        rewards = []
        truncations = []
        dones = []
        infos = []
        logic_states = []
        neural_states = []
        for i, env in enumerate(self.envs):
            action = actions[i]
            # make a step in the env
            obs, reward, truncation, done, info = env.step(action)
            # lazy frame to tensor
            obs = torch.tensor(obs).float()
            # get logic and neural state
            state = env.objects
            raw_state = obs
            logic_state, neural_state = self.convert_state(state, raw_state)
            logic_states.append(logic_state)
            neural_states.append(neural_state)
            observations.append(obs)
            rewards.append(reward)
            truncations.append(truncation)
            dones.append(done)
            infos.append(info)
            # store final info
            
            
        # observations = torch.stack(observations)
        return (torch.stack(logic_states), torch.stack(neural_states)), rewards, truncations, dones, infos

            
        # # obs, reward, done, truncations, infos = self.env.step(action)
        # obs, reward, truncations, done, infos = self.env.step(action)
        
        # # ste RGB env
        # # x = self.raw_env.step(action.unsqueeze(0)) 
        # # raw_obs, raw_reward, raw_done, _, _ = x
        # # assert reward == raw_reward and done == raw_done, "Two envs conflict: rewards: {} and {}, dones: {} and {}".format(reward, raw_reward, done, raw_done)  
        # # assert done == raw_done, "Two envs conflict: dones: {} and {}".format(done, raw_done)  
        # state = self.env.objects
        # raw_state = obs #self.env.dqn_obs
        # # raw_state = raw_obs
        # # raw_state = raw_state.unsqueeze(0)
        # logic_state, neural_state = self.convert_state(state, raw_state)
        # # if len(logic_state.shape) == 2:
        # logic_state = logic_state.unsqueeze(0)
        # return (logic_state, neural_state), reward, done, truncations, infos

    def extract_logic_state(self, input_state):
        # state_list = []
        # for input_state in input_states:
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)

        obj_count = {k: 0 for k in MAX_ESSENTIAL_OBJECTS.keys()}

        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "OxygenBar":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = obj.orientation.value if obj.orientation is not None else 0
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
        return state
        #     states.append(state)
        # states = torch.stack(state_list)
        # return states

    def extract_neural_state(self, raw_input_state):
        return raw_input_state
        # return torch.tensor(raw_input_state).float()
        # print(raw_input_state.shape)
        # return torch.Tensor(raw_input_state)#.unsqueeze(0)#.float()

    def close(self):
        self.env.close()
