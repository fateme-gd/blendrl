import torch

from nsfr.common import get_nsfr_model, get_meta_nsfr_model
from nsfr.utils.common import load_module



def get_meta_actor(env, meta_rules, device, train=True, mode='logic'):
    assert mode in ['logic', 'neural']
    if mode == 'logic':
        return get_meta_nsfr_model(env.name, meta_rules, device, train=train)
    if mode == 'neural':
        mlp_module_path = f"in/envs/{env.name}/mlp.py"
        module = load_module(mlp_module_path)
        return module.MLP(out_size=1, has_sigmoid=True, device=device)
    
    
def extract_policy_probs(NSFR, V_T, device):
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