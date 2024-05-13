import random
import pickle
from pathlib import Path
import os


import torch
import torch.nn as nn
import torch.nn.functional as F 
from .logic_agent import LogicPPO, NsfrActorCritic
from .neural_agent import NeuralPPO, ActorCritic
from neumann.src.torch_utils import softor
# from nudge.env import NudgeBaseEnv
from torch.distributions.categorical import Categorical


from torch.distributions import Categorical
from nsfr.utils.common import load_module
from nsfr.common import get_nsfr_model

from utils import get_meta_actor, extract_policy_probs, load_pretrained_stable_baseline_ppo, load_cleanrl_agent

from stable_baselines3 import PPO

class DeicticActor(nn.Module):
    def __init__(self, env, neural_actor, logic_actor, meta_actor, actor_mode, meta_mode, device=None):
        super(DeicticActor, self).__init__()
        self.env = env
        self.neural_actor = neural_actor
        self.logic_actor = logic_actor
        self.meta_actor = meta_actor
        self.actor_mode = actor_mode
        self.meta_mode = meta_mode
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
            exist_flag = False
            for j,action_pred_name in enumerate(self.logic_actor.get_prednames()):
                if env_action_name in action_pred_name:
                    #if i not in env_action_id_to_action_pred_indices:
                    #    env_action_id_to_action_pred_indices[i] = []
                    env_action_id_to_action_pred_indices[i].append(j)
                    exist_flag = True
            if not exist_flag:
                # i-th env action is not defined by any rules thus will be always 0.0
                # refer to dummy predicte index
                # pred1, pred2, ..., predn, dummy_pred
                dummy_index = len(self.logic_actor.get_prednames())
                env_action_id_to_action_pred_indices[i].append(dummy_index)

                
        return env_action_id_to_action_pred_indices
        
    def compute_action_probs_hybrid(self, neural_state, logic_state):
        # logic_action_probs = self.logic_a2c.actor(logic_state)
        # neural_action_probs = self.neural_a2c.actor(neural_state)
        # state: B * N
        batch_size = neural_state.size(0)
        logic_action_probs = self.to_action_distribution(self.logic_actor(logic_state))
        # neural_action_probs = self.neural_actor(neural_state)
        neural_action_probs = self.to_neural_action_distribution(neural_state)
        
        # action_probs: B * N_actions
        # beta = self.switch(neural_state)
        batch_size = neural_state.size(0)
        # beta = torch.tensor([[0.5]]).repeat(batch_size, 1).to(self.device)
        # beta = torch.tensor([[1.0]]).repeat(batch_size, 1).to(self.device)
        # ones = torch.ones_like(beta).to(self.device)
        
        # B * 2
        weights = self.to_meta_policy_distribution(neural_state, logic_state)
        # save weights
        self.w_policy = weights[0]
        
        # print("neural: {}, logic: {}".format(weights[:,0], weights[:,1]))
        n_actions = neural_action_probs.size(1)
        
        # B * N_actions * 2
        weights = weights.unsqueeze(1).repeat(1, n_actions, 1)  
        
        # print(weights)
        
        # p = w1 * p_neural + w2 * p_logic
        
        action_probs = weights[:,:,0] * neural_action_probs + weights[:,:,1] * logic_action_probs
        # merge action probs 
        # merged_values = softor([logic_action_probs, neural_action_probs], dim=1)
        # action_probs = torch.softmax(merged_values, dim=0)
        return action_probs
    
    def compute_action_probs_logic(self, logic_state):
        self.w_policy = torch.tensor([0.0, 1.0], device=self.device)
        logic_action_probs = self.to_action_distribution(self.logic_actor(logic_state))
        return logic_action_probs
    
    def compute_action_probs_neural(self, neural_state):
        self.w_policy = torch.tensor([1.0, 0.0], device=self.device)
        neural_action_probs = self.to_neural_action_distribution(neural_state)
        return neural_action_probs


    

    # def select_policy(self, neural_state, logic_state):
    #     V_T_meta = self.meta_actor(logic_state)
    #     policy_probs = self.to_policy_distribution(V_T_meta)
    #     return policy_probs
        # policy_select_vector = F.gumbel_softmax(policy_probs)
        # return policy_select_vector
        
        
    def to_meta_policy_distribution(self, neural_state, logic_state):
        # get prob for neural and logic policy
        # probs = extract_policy_probs(self.meta_actor, V_T, self.device)
        # to logit
        assert self.meta_mode in ['logic', 'neural'], "Invalid meta mode {}".format(self.meta_mode)
        if self.meta_mode == 'logic':
            policy_probs = self.meta_actor(logic_state)
        else:
            policy_probs = self.meta_actor(neural_state)
            
        logits = torch.logit(policy_probs, eps=0.01)
        # return torch.softmax(logits, dim=1)
        return F.gumbel_softmax(logits, dim=1)
        # take softmax
        # dist = torch.softmax(logits, dim=1)
        # return dist
    
    
    def to_action_distribution(self, raw_action_probs):
        """Converts raw action probabilities to a distribution."""
        
        
        batch_size = raw_action_probs.size(0)
        env_action_names = list(self.env.pred2action.keys())        
        
        # action_probs = torch.zeros(len(env_action_names))

        # TODO: put dummy value to TAIL in case of no action predicate
        raw_action_probs = torch.cat([raw_action_probs, torch.zeros(batch_size, 1, device=self.device)], dim=1)
        dist_values = []
        for i in range(len(env_action_names)):
            if i in self.env_action_id_to_action_pred_indices:
                indices = torch.tensor(self.env_action_id_to_action_pred_indices[i], device=self.device)\
                    .expand(batch_size, -1).to(self.device)
                gathered = torch.gather(raw_action_probs, 1, indices)
                # merged value for i-th action for samples in the batch
                merged = softor(gathered, dim=1) # (batch_size, 1) 
                dist_values.append(merged)
        
        
        action_values = torch.stack(dist_values,dim=1) # (batch_size, n_actions) 
                
                
        # action_raw_dist = torch.stack([softor(action_values, dim=1) for action_values in dist_values])
        action_dist = torch.softmax(action_values, dim=1)
        
        # if action_dist.size(1) < self.env.n_raw_actions:
        #     zeros = torch.zeros(batch_size, self.env.n_raw_actions - action_dist.size(1), device=self.device, requires_grad=True)
        #     action_dist = torch.cat([action_dist, zeros], dim=1)
        action_dist = self.reshape_action_distribution(action_dist)
        return action_dist
        
    def to_neural_action_distribution(self, neural_state):
        # actions, values, log_prob = self.neural_actor(neural_state)
        # action_dist = torch.softmax(values, dim=1)
        
        # action_dist = self.reshape_action_distribution(action_dist)
        # action_dist = self.neural_actor.get_distribution(neural_state).distribution.probs
        hidden = self.neural_actor.network(neural_state / 255.0)
        logits = self.neural_actor.actor(hidden)
        probs = Categorical(logits=logits)
        action_dist = probs.probs
        return action_dist
    
    def reshape_action_distribution(self, action_dist):
        batch_size = action_dist.size(0)
        if action_dist.size(1) < self.env.n_raw_actions:
            zeros = torch.zeros(batch_size, self.env.n_raw_actions - action_dist.size(1), device=self.device, requires_grad=True)
            action_dist = torch.cat([action_dist, zeros], dim=1)
        return action_dist
    
    def forward(self, neural_state, logic_state):
        assert self.actor_mode in ["hybrid", "logic", "neural"], "Invalid actor mode {}".format(self.actor_mode) 
        if self.actor_mode == "hybrid":
            return self.compute_action_probs_hybrid(neural_state, logic_state)
        elif self.actor_mode == "logic":
            return self.compute_action_probs_logic(logic_state)
        else:
            return self.compute_action_probs_neural(neural_state)
    
        

class DeicticActorCritic(nn.Module):
    def __init__(self, env, rules, actor_mode, meta_mode, device, rng=None):
        super(DeicticActorCritic, self).__init__()
        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.actor_mode = actor_mode
        self.meta_mode = meta_mode
        # self.neural_a2c = ActorCritic(env, device=device)
        # self.logic_a2c = NsfrActorCritic(env, rules, device=device)
        self.env = env
        self.rules = rules
        mlp_module_path = f"in/envs/{self.env.name}/mlp.py"
        module = load_module(mlp_module_path)
        # self.neural_actor = module.MLP(has_softmax=True, device=device)
        # self.baseline_ppo = PPO("MlpPolicy", env.raw_env, verbose=1)
        # self.baseline_ppo = load_pretrained_stable_baseline_ppo(env, device)
        # self.visual_neural_actor = self.baseline_ppo.policy #.mlp_extractor.policy_net
        # self.visual_neural_actor, self.critic = load_cleanrl_agent(env=env.raw_env, pretrained=False, device=device)
        self.visual_neural_actor = load_cleanrl_agent(pretrained=False, device=device)
        
        self.logic_actor = get_nsfr_model(env.name, rules, device=device, train=True)
        self.meta_actor = get_meta_actor(env, rules, device, train=True, meta_mode=meta_mode)#train=False)
        # self.meta_actor = module.MLP(out_size=1, has_sigmoid=True, device=device)
        self.actor = DeicticActor(env, self.visual_neural_actor, self.logic_actor, self.meta_actor, actor_mode, meta_mode, device=device)
        # self.critic = module.MLP(device=device, out_size=1, logic=True)
        # self.critic = self.baseline_ppo.policy.mlp_extractor.value_net
        # self.critic.to(device)
        
        # the number of actual actions on the environment
        self.num_actions = len(self.env.pred2action.keys())
        
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions-1) for _ in range(self.num_actions-1)], device=device))
        

    def _print(self):
        if self.meta_mode == 'logic':
            print("==== Meta Policy ====")
            self.meta_actor.print_program()
        print("==== Logic Policy ====")
        self.logic_actor.print_program()
        
    def get_policy_weights(self):
        return self.actor.w_policy
    
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
        # state_values = self.critic(neural_state)
        state_values = self.visual_neural_actor.get_value(neural_state)

        return action_logprobs, state_values, dist_entropy

    def get_prednames(self):
        return self.actor.get_prednames()
    
    def get_action_and_value(self, neural_state, logic_state, action=None):
        # compute action
        action_probs = self.actor(neural_state, logic_state)
        dist = Categorical(action_probs)
        if action is None:
            action = dist.sample()
        # action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.device)
        # if torch.numel(action) > 1:
        #     action = action[0]
        logprob = dist.log_prob(action)
        
        # compute value
        value = self.visual_neural_actor.get_value(neural_state)
        
        # action, action_logprob = self.act(neural_state, logic_state, epsilon=epsilon)
        print(action, dist.probs)
        return action, logprob, dist.entropy(), value
    
    def get_value(self, neural_state, logic_state):
        value = self.visual_neural_actor.get_value(neural_state)
        return value
    
    def save(self, checkpoint_path, directory: Path, step_list, reward_list, weight_list):
        torch.save(self.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)
            pickle.dump(weight_list, f)


class DeicticPPO(nn.Module):
    # def __init__(self, env: NudgeBaseEnv, rules: str, lr_actor, lr_critic, optimizer,
    #              gamma, epochs, eps_clip, device=None):
    def __init__(self, env, rules, lr_actor, lr_critic, optimizer, gamma, epochs, eps_clip, actor_mode, meta_mode, device):
        super(DeicticPPO, self).__init__()
        self.device = device
        self.actor_mode = actor_mode
        self.meta_mode = meta_mode
        # self.logic_ppo = LogicPPO(*logic_ppo_params)
        # self.neural_ppo = NeuralPPO(*neural_ppo_params)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.buffer = RolloutBuffer()
        self.policy = DeicticActorCritic(env, rules, actor_mode, meta_mode, device)
        # self.optimizer = optimizer([
        #     # {'params': self.policy.logic_actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.meta_actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     # {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])
        self.optimizer = optimizer(list(self.parameters()))

        self.policy_old = DeicticActorCritic(env, rules, actor_mode, meta_mode, device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.prednames = self.get_prednames()

        self.MseLoss = nn.MSELoss()
        # self._freeze_neural_actor()
        
    # def get_action_and_value(next_obs):
    #     neural_obs, logic_obs = next_obs
    
    def select_action(self, state, epsilon=0.0):
        logic_state, neural_state = state
        logic_state = torch.tensor(logic_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        neural_state = torch.tensor(neural_state, dtype=torch.float32, device=self.device)#.unsqueeze(0)

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

        return action
        # predicate = self.prednames[action.item()]
        # return predicate

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

        # old_neural_states = torch.squeeze(torch.stack(self.buffer.neural_states, dim=0)).detach().to(self.device)
        old_neural_states = torch.squeeze(torch.stack(self.buffer.neural_states, dim=0), dim=1).detach().to(self.device)
        # print(old_neural_states.size())
        old_logic_states = torch.squeeze(torch.stack(self.buffer.logic_states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        total_loss = 0
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
            total_loss += loss.mean().item()
            # for name, param in self.policy.named_parameters():
            #     print(name, param.grad)
            self.optimizer.step()
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        avg_loss = total_loss / self.epochs
        return avg_loss
        
        
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
        return self.policy.logic_actor.get_prednames()
    
    def _freeze_neural_actor(self):
        for param in self.policy.visual_neural_actor.parameters():
            param.requires_grad = False
        for param in self.policy_old.visual_neural_actor.parameters():
            param.requires_grad = False


        
        

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
