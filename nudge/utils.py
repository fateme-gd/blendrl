import math
import random
import numpy as np
import torch
import yaml
from pathlib import Path
import os
import re

from .agents.logic_agent import NsfrActorCritic
from .agents.neural_agent import ActorCritic
from nudge.env import NudgeBaseEnv
from functools import reduce
from nsfr.utils.torch import softor

 
def to_proportion(dic):
    # Using reduce to get the sum of all values in the dictionary
    temp = reduce(lambda x, y: x + y, dic.values())
 
    # Using dictionary comprehension to divide each value by the sum of all values
    res = {k: v / temp for k, v in dic.items()}
    return res

def get_action_stats(env, actions):
    env_actions = env.pred2action.keys()
    frequency_dic = {}
    for action in env_actions:
        frequency_dic[action] = 0
        
    for i, action in enumerate(actions):
        frequency_dic[action] += 1
    
    action_proportion = to_proportion(frequency_dic)
    return action_proportion

def save_hyperparams(signature, local_scope, save_path, print_summary: bool = False):
    hyperparams = {}
    for param in signature.parameters:
        hyperparams[param] = local_scope[param]
    with open(save_path, 'w') as f:
        yaml.dump(hyperparams, f)
    if print_summary:
        print("Hyperparameter Summary:")
        print(open(save_path).read())


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_noise(obj, index_obj, num_of_objs):
    mean = torch.tensor(0.2)
    std = torch.tensor(0.05)
    noise = torch.abs(torch.normal(mean=mean, std=std)).item()
    rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
    rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
    rand_noises.insert(index_obj, 1 - noise)

    for i, noise in enumerate(rand_noises):
        obj[i] = rand_noises[i]
    return obj


def simulate_prob(extracted_states, num_of_objs, key_picked):
    for i, obj in enumerate(extracted_states):
        obj = add_noise(obj, i, num_of_objs)
        extracted_states[i] = obj
    if key_picked:
        extracted_states[:, 1] = 0
    return extracted_states


def load_model(model_dir,
               env_kwargs_override: dict = None,
               device=torch.device('cuda:0')):
    from .agents.blender_agent import BlenderActorCritic
    # Determine all relevant paths
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    checkpoint_dir = model_dir / "checkpoints"
    most_recent_step = get_most_recent_checkpoint_step(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"step_{most_recent_step}.pth"

    # Load model's configuration
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    algorithm = config["algorithm"]
    environment = config["environment"]
    env_kwargs = config["env_kwargs"]
    env_kwargs.update(env_kwargs_override)

    # Setup the environment
    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **env_kwargs)

    rules = config["rules"]

    print("Loading...")
    # Initialize the model
    if algorithm == 'ppo':
        model = ActorCritic(env).to(device)
    elif algorithm == 'logic':
        model = NsfrActorCritic(env, device=device, rules=rules).to(device)
    else:
        model = BlenderActorCritic(env, rules=rules, actor_mode=config["actor_mode"], blender_mode=config["blender_mode"], device=device).to(device)

    # Load the model weights
    with open(checkpoint_path, "rb") as f:
        model.load_state_dict(state_dict=torch.load(f, map_location=torch.device('cpu')))
    # model.logic_actor.im.W = torch.nn.Parameter(model.logic_actor.im.init_identity_weights(device))
    # print(model.logic_actor.im.W)

    return model


def yellow(text):
    return "\033[93m" + text + "\033[0m"


def exp_decay(episode: int):
    """Reaches 2% after about 850 episodes."""
    return max(math.exp(-episode / 500), 0.02)


def get_most_recent_checkpoint_step(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    highest_step = 0
    pattern = re.compile("[0-9]+")
    for i, c in enumerate(checkpoints):
        match = pattern.search(c)
        if match is not None:
            step = int(match.group())
            if step > highest_step:
                highest_step = step
    return highest_step


def print_program(agent, mode="softor"):
    """Print a summary of logic programs using continuous weights."""
    try:
        nsfr = agent.policy.actor
    except AttributeError:
        try:
            nsfr = agent.actor
        except AttributeError:
            nsfr = agent
    if mode == "argmax":
        C = nsfr.clauses
        Ws_softmaxed = torch.softmax(nsfr.im.W, 1)
        for i, W_ in enumerate(Ws_softmaxed):
            max_i = np.argmax(W_.detach().cpu().numpy())
            print('C_' + str(i) + ': ',
                  C[max_i], 'W_' + str(i) + ':', round(W_[max_i].detach().cpu().item(), 3))
    elif mode == "softor":
        W_softmaxed = torch.softmax(nsfr.im.W, 1)
        w = softor(W_softmaxed, dim=0)
        for i, c in enumerate(nsfr.clauses):
            print('C_' + str(i) + ': ', np.round(w[i].detach().cpu().item(), 2), nsfr.clauses[i])
            
