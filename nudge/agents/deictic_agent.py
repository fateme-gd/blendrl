import random
import pickle
from pathlib import Path
import os


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
    def __init__(self, env, neural_actor, logic_actor, switch, device=None):
        super(DeicticActor, self).__init__()
        self.env = env
        self.neural_actor = neural_actor
        self.logic_actor = logic_actor
        self.switch = switch
        self.device = device
        self.env_action_id_to_action_pred_indices = self._build_action_id_dict()
        
    def _build_action_id_dict(self):
        env_action_names = list(self.env.pred2action.keys())        
        # action_probs = torch.zeros(len(env_action_names))
        env_action_id_to_action_pred_indices = {}
        # init dic
        for i, env_action_name in enumerate(env_action_names):
            env_action_id_to_action_pred_indices[i] = []
            
        for i, env_action_name in enumerate(env_action_names):
            for j,action_pred_name in enumerate(self.logic_actor.get_prednames()):
                if env_action_name in action_pred_name:
                    #if i not in env_action_id_to_action_pred_indices:
                    #    env_action_id_to_action_pred_indices[i] = []
                    env_action_id_to_action_pred_indices[i].append(j)
                    # torch.tensor([0.0], device=self.device)
        return env_action_id_to_action_pred_indices
        
    def compute_action_probs(self, neural_state, logic_state):
        # logic_action_probs = self.logic_a2c.actor(logic_state)
        # neural_action_probs = self.neural_a2c.actor(neural_state)
        logic_action_probs = self.to_action_distribution(self.logic_actor(logic_state))
        neural_action_probs = self.neural_actor(neural_state)
        beta = self.switch(neural_state)
        ones = torch.ones_like(beta).to(self.device)
        
        action_probs = beta * neural_action_probs + (ones - beta) * logic_action_probs
        # merge action probs 
        # merged_values = softor([logic_action_probs, neural_action_probs], dim=1)
        # action_probs = torch.softmax(merged_values, dim=0)
        return action_probs
    
    def to_action_distribution(self, raw_action_probs):
        """Converts raw action probabilities to a distribution."""
        #TODO: Implement this method
        
        
        batch_size = raw_action_probs.size(0)
        env_action_names = list(self.env.pred2action.keys())        
        
        # action_probs = torch.zeros(len(env_action_names))
                
        dist_values = []
        for i in range(len(env_action_names)):
            if i in self.env_action_id_to_action_pred_indices:
                indices = torch.tensor(self.env_action_id_to_action_pred_indices[i], device=self.device)\
                    .expand(batch_size, -1)
                gathered = torch.gather(raw_action_probs, 1, indices)
                # merged value for i-th action for samples in the batch
                merged = softor(gathered, dim=1) # (batch_size, 1) 
                dist_values.append(merged)
        
        
        action_values = torch.stack(dist_values,dim=1) # (batch_size, n_actions) 
                
                
        # action_raw_dist = torch.stack([softor(action_values, dim=1) for action_values in dist_values])
        action_dist = torch.softmax(action_values, dim=1)
        return action_dist
        
    
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
        self.switch = module.MLP(out_size=1, has_sigmoid=True)
        self.actor = DeicticActor(env, self.neural_actor, self.logic_actor, self.switch)
        self.critic = module.MLP(out_size=1, logic=True)
        
        # the number of actual actions on the environment
        self.num_actions = len(self.env.pred2action.keys())
        
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions-1) for _ in range(self.num_actions-1)], device=device))

        
    def forward(self):
        raise NotImplementedError
    
    def act(self, neural_state, logic_state, epsilon=0.0):
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
        action_probs = self.actor(neural_state, logic_state)   
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
        self.gamma = self.logic_ppo.gamma
        self.eps_clip = self.logic_ppo.eps_clip
        self.epochs = self.logic_ppo.epochs
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

        predicate = self.logic_ppo.prednames[action.item()]
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
            # for name, param in self.policy.named_parameters():
            #     print(name, param.grad)
            self.optimizer.step()
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        
    def save(self, checkpoint_path, directory: Path, step_list, reward_list, weight_list):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)

    def load(self, directory: Path):
        self.neural_ppo.load(directory)
        self.logic_ppo.load(directory)
        # only for recover form crash
        model_name = input('Enter file name: ')
        model_file = os.path.join(directory, model_name)
        self.policy_old.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        with open(directory / "data.pkl", "rb") as f:
            step_list = pickle.load(f)
            reward_list = pickle.load(f)
            weight_list = pickle.load(f)
        return step_list, reward_list, weight_list

    def get_predictions(self, state):
        self.prediction = state
        return self.prediction

    def get_weights(self):
        return self.policy.actor.get_params()

    def get_prednames(self):
        return self.policy.actor.get_prednames()


        
        

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
