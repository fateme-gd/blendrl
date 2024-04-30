from typing import Sequence

from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np
import torch as th
from ocatari.ram.seaquest import MAX_ESSENTIAL_OBJECTS
import gymnasium

class NudgeEnv(NudgeBaseEnv):
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

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False, seed=None):
        super().__init__(mode)
        self.raw_env = gymnasium.make("SeaquestNoFrameskip-v4")
        self.env = OCAtari(env_name="SeaquestNoFrameskip-v4", mode="ram",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)
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
        _, _ = self.env.reset(seed=self.seed)
        raw_state, _ = self.raw_env.reset(seed=self.seed)
        state = self.env.objects
        return self.extract_logic_state(state), self.extract_neural_state(raw_state)
        # return  self.convert_state(state, raw_state)

    def step(self, action, is_mapped: bool = False):
        # if not is_mapped:
        #     action = self.map_action(action)
        # step RAM env
        obs, reward, done, _, _ = self.env.step(action)
        # ste RGB env
        raw_obs, raw_reward, raw_done, _, _ = self.raw_env.step(action)
        assert reward == raw_reward and done == raw_done
        state = self.env.objects
        raw_state = raw_obs
        return self.convert_state(state, raw_state), reward, done

    def extract_logic_state(self, input_state):
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

    def extract_neural_state(self, raw_input_state):
        return raw_input_state

    def close(self):
        self.env.close()
