import random

import torch
import torch.nn as nn

from .logic_agent import LogicPPO, NsfrActorCritic
from .neural_agent import NeuralPPO, ActorCritic
from neumann.src.torch_utils import softor
# from nudge.env import NudgeBaseEnv

from torch.distributions import Categorical
from nsfr.utils.common import load_module
from nsfr.common import get_nsfr_model

class DeicticActor(nn.Module):
    def __init__(self, env, neural_actor, logic_actor, device=None):
        super(DeicticActor, self).__init__()
        self.env = env
        self.neural_actor = neural_actor
        self.logic_actor = logic_actor
        self.device = device
    
    def compute_action_probs(self, neural_state, logic_state):
        # logic_action_probs = self.logic_a2c.actor(logic_state)
        # neural_action_probs = self.neural_a2c.actor(neural_state)
        logic_action_probs = self.to_action_distribution(self.logic_actor(logic_state))
        neural_action_probs = self.neural_actor(neural_state)
        # merge action probs 
        action_probs = torch.softmax(softor([logic_action_probs, neural_action_probs]))
        return action_probs
    
    def to_action_distribution(self, raw_action_probs):
        """Converts raw action probabilities to a distribution."""
        #TODO: Implement this method
        
        
        env_action_names = list(self.env.pred2action.keys())
        # pred_to_action_index = []
        # for action_atom in action_atoms:
        #     action_pred_name = action_atom.predicate.name
        #     for i, env_action in enumerate(env_action_names):
        #         if env_action in action_pred_name:
        #             pred_to_action_index.append()
        #             break
        # index_tensor = torch.tensor(pred_to_action_index)  
        # dist = torch.gather(raw_action_probs, 1, index_tensor)     
        # model_action: right_to_diver
        # pred_name: right
        
        # action_probs = torch.zeros(len(env_action_names))
        env_action_id_to_action_pred_indices = {}
        for i, env_action_name in enumerate(env_action_names):
            for j,action_pred_name in enumerate(self.logic_actor.get_prednames()):
                if env_action_name in action_pred_name:
                    if i not in env_action_id_to_action_pred_indices:
                        env_action_id_to_action_pred_indices[i] = []
                    env_action_id_to_action_pred_indices[i].append(j)
                
        dist_values = []
        for i in range(len(env_action_names)):
            if i in env_action_id_to_action_pred_indices:
                indices = torch.tensor(env_action_id_to_action_pred_indices[i], device=self.device)
                dist_values.append(torch.gather(raw_action_probs[0], 0, indices))
                
                
        action_dist = torch.stack([softor(action_values) for action_values in dist_values])
        return action_dist
        #     probs = []
        #     for index in indices:
        #         probs.append(raw_action_probs[index])
        #     dist_values.append(softor(probs))
        
        
        # merged_env_action_distribution = []
        # for env_action in env_action_names:
        #     probs_for_an_env_action = []
        #     for i, action_atom in enumerate(action_atoms):
        #         if env_action in action_atom.predicate.name:
        #             probs_for_an_env_action.append(raw_action_probs[i])
        #     merged_prob_for_an_env_action = softor(probs_for_an_env_action)
        #     merged_env_action_distribution.append(merged_prob_for_an_env_action)
        # action_distribution = torch.cat(merged_env_action_distribution)
        # return action_distribution
        
    
    def forward(self, neural_state, logic_state):
        return self.compute_action_probs(neural_state, logic_state)
        

class DeicticActorCritic(nn.Module):
    def __init__(self, env, rules, device=None, rng=None):
        super(DeicticActorCritic, self).__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        
        # self.neural_a2c = ActorCritic(env, device=device)
        # self.logic_a2c = NsfrActorCritic(env, rules, device=device)
        self.env = env
        self.rules = rules
        mlp_module_path = f"in/envs/{self.env.name}/mlp.py"
        module = load_module(mlp_module_path)
        self.neural_actor = module.MLP(has_softmax=True)
        self.logic_actor = get_nsfr_model(env.name, rules, device=device, train=True)
        self.actor = DeicticActor(env, self.neural_actor, self.logic_actor)
        self.critic = module.MLP(out_size=1, logic=True)
        
        
    def forward(self):
        raise NotImplementedError
    
    # def compute_action_probs(self, neural_state, logic_state):
    #     # logic_action_probs = self.logic_a2c.actor(logic_state)
    #     # neural_action_probs = self.neural_a2c.actor(neural_state)
    #     logic_action_probs = self.logic_actor(logic_state)
    #     neural_action_probs = self.neural_actor(neural_state)
    #     # merge action probs 
    #     action_probs = torch.softmax(softor([logic_action_probs, neural_action_probs]))
    #     return action_probs

    def act(self, neural_state, logic_state, epsilon=0.0):
        # logic_action_probs = self.logic_a2c.actor(logic_state)
        # neural_action_probs = self.neural_a2c.actor(neural_state)
        # merge action probs 
        # action_probs = self.compute_action_probs(neural_state, logic_state)
        action_probs = self.actor(neural_state, logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.device)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, neural_state, logic_state, action):
        action_probs = self.compute_action_probs(neural_state, logic_state)   
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(neural_state)

        return action_logprobs, state_values, dist_entropy

    def get_prednames(self):
        return self.actor.get_prednames()
    

class DeicticPPO:
    # def __init__(self, env: NudgeBaseEnv, rules: str, lr_actor, lr_critic, optimizer,
    #              gamma, epochs, eps_clip, device=None):
    def __init__(self, env, neural_ppo_params, logic_ppo_params, rules, optimizer, lr_actor, lr_critic, device=None):
        self.device = device
        self.logic_ppo = LogicPPO(*logic_ppo_params)
        self.neural_ppo = NeuralPPO(*neural_ppo_params)
        self.buffer = RolloutBuffer()
        self.policy = DeicticActorCritic(env, rules, device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = DeicticActorCritic(env, rules, device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, epsilon=0.0):
        logic_state, neural_state = state
        logic_state = torch.tensor(logic_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        neural_state = torch.tensor(neural_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            # import ipdb; ipdb.set_trace()
            # logic_action, logic_action_logprob = self.logic_ppo.policy_old.act(logic_state, epsilon=epsilon)
            # neural_action, neural_action_logprob = self.neural_ppo.policy_old.act(neural_state, epsilon=epsilon)
            action, action_logprob = self.policy_old.act(neural_state, logic_state, epsilon=epsilon)
        
        self.buffer.neural_states.append(neural_state)
        self.buffer.logic_states.append(logic_state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        predicate = self.prednames[action.item()]
        return predicate

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor

        old_neural_states = torch.squeeze(torch.stack(self.buffer.neural_states, dim=0)).detach().to(self.device)
        old_logic_states = torch.squeeze(torch.stack(self.buffer.logic_states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_neural_states, old_logic_states,
                                                                        old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # training does not converge if the entropy term is added ...
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)  # - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.neural_states = []
        self.logic_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.neural_states[:]
        del self.logic_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]
