import torch
import gymnasium as gym
from nsfr.common import get_nsfr_model, get_meta_nsfr_model
from nsfr.utils.common import load_module

from stable_baselines3 import PPO
# from huggingface_sb3 import load_from_hub, push_to_hub


def get_meta_actor(env, meta_rules, device, train=True, mode='logic'):
    assert mode in ['logic', 'neural']
    if mode == 'logic':
        return get_meta_nsfr_model(env.name, meta_rules, device, train=train)
    if mode == 'neural':
        mlp_module_path = f"in/envs/{env.name}/mlp.py"
        module = load_module(mlp_module_path)
        return module.MLP(out_size=1, has_sigmoid=True, device=device)
    
    
def extract_policy_probs(NSFR, V_T, device):
    """not to be used. meta actor already computes this in the forward function."""
    batch_size = V_T.size(0)
    # extract neural_agent(img) and logic_agent(image)
    indices = []
    for i, atom in enumerate(NSFR.atoms):
        if "neural_agent" in str(atom):
            indices.append(i)
    for i, atom in enumerate(NSFR.atoms):
        if "logic_agent" in str(atom):
            indices.append(i)
    
    indices = torch.tensor(indices).to(device).unsqueeze(0)
    indices = indices.repeat(batch_size, 1)
    
    policy_probs = torch.gather(V_T, 1, indices)
    return policy_probs


def load_pretrained_stable_baseline_ppo(env, device):
    agent_path = "rl-baselines3-zoo/rl-trained-agents/ppo/SeaquestNoFrameskip-v4_1/SeaquestNoFrameskip-v4.zip"
    policy_path = "rl-baselines3-zoo/rl-trained-agents/ppo/SeaquestNoFrameskip-v4_1/policy.pth"
    policy = torch.load(policy_path) #, map_location=device)
    
    # checkpoint = load_from_hub("ThomasSimonini/ppo-SeaquestNoFrameskip-v4", "ppo-SeaquestNoFrameskip-v4.zip")

    # # Because we using 3.7 on Colab and this agent was trained with 3.8 to avoid Pickle errors:
    # custom_objects = {
    #         "learning_rate": 0.0,
    #         "lr_schedule": lambda _: 0.0,
    #         "clip_range": lambda _: 0.0,
    #     }
    # baseline_ppo = PPO.load(checkpoint, custom_objects=custom_objects)

    # saved_variables = torch.load(agent_path, map_location=device, weights_only=False)

    # baseline_ppo = PPO("MlpPolicy", env.raw_env, verbose=1)
    # saved_variables = torch.load(agent_path, map_location=device)
    # saved_variables = torch.load(agent_path, map_location=device)
    # baseline_ppo.load_state_dict(saved_variables["state_dict"])
    # baseline_ppo.to(device)
    # return baseline_ppo
    # Create policy object
    # model = cls(**saved_variables["data"])
    # Load weights
    # model.load_state_dict(saved_variables["state_dict"])
    
    # baseline_ppo = PPO("MlpPolicy", env.raw_env, verbose=1)
    baseline_ppo = PPO.load(agent_path, env=env.raw_env, device=device)
    return baseline_ppo
    # neural_pixel_actor =baseline_ppo.policy #.mlp_extractor.policy_net
    # base_path = "rl-baselines3-zoo/rl-trained-agents/a2c/SeaquestNoFrameskip-v4_1"
    
    
def load_cleanrl_envs(env_id, run_name=None, capture_video=False, num_envs=1):
    from cleanrl.cleanrl.ppo_atari import make_env
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, capture_video, run_name) for i in range(num_envs)],
    )
    return envs
    
def load_cleanrl_agent(envs, device):
    from cleanrl.cleanrl.ppo_atari import Agent
    agent = Agent(envs) #, device=device, verbose=1)
    agent.load_state_dict(torch.load("cleanrl/out/ppo_Seaquest-v4_1.pth"))
    agent.to(device)
    return agent, agent