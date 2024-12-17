
import pickle
import pandas as pd

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    # print("Is Nan: ", pd.isnull(scalars).any())
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        

    # print("smoothed: ", smoothed)
    return pd.DataFrame(smoothed)


def load_data(env_name, model, max_episodes=25000, initial_skip=0):
    episodic_returns_list = []
    episodic_lengths_list = []
    value_losses_list = []
    policy_losses_list = []
    entropies_list = []
    blend_entropies_list = []
    
    # models = ['blendrl', 'nudge', 'neuralppo']
    # for model in models:
    model_base_folder = "./runs/{}/".format(model)
        
    for i in range(3):
        folder_name =  env_name + "_{}".format(i)
        file_path = model_base_folder + folder_name + "/checkpoints/training_log.pkl"

        with open(file_path, "rb") as f:
                episodic_returns, episodic_lengths, value_losses, policy_losses, entropies, blend_entropies = pickle.load(f)
                # append to list 
                episodic_returns_list.append(episodic_returns)
                episodic_lengths_list.append(episodic_lengths)
                value_losses_list.append(value_losses)
                policy_losses_list.append(policy_losses)
                entropies_list.append(entropies)
                blend_entropies_list.append(blend_entropies)
                
    # smoothing
    df_rewards = pd.DataFrame(episodic_returns_list).astype(float).T[:max_episodes]
    df_rewards_smooth = []
    for _, rewards in df_rewards.items():
        # print("Rewards: ", rewards)
        df_rewards_smooth.append(smooth(rewards, 0.99))
    df_rewards_smooth = pd.concat(df_rewards_smooth, axis=1)
    
    # compute mean and stds
    mean = df_rewards_smooth.mean(axis=1)
    std = df_rewards_smooth.std(axis=1)
                
    return mean, std

def load_csv(env_name, model, max_episodes=15000, scale_factor=1000):
    df_rewards_all = pd.read_csv(f'csv/{model}_{env_name}.csv') * scale_factor
    df_rewards = pd.concat([df_rewards_all.iloc[:,1].dropna(), df_rewards_all.iloc[:,4].dropna(), df_rewards_all.iloc[:,7].dropna()], axis=1)[:max_episodes]
    df_rewards.columns = ['0', '1', '2']
    
    df_rewards_smooth = []
    for _, rewards in df_rewards.items():
        # print("Rewards: ", rewards)
        rewards = rewards.interpolate() # To remove NaN values
        df_rewards_smooth.append(smooth(rewards, 0.99))
    df_rewards_smooth = pd.concat(df_rewards_smooth, axis=1)
    
    # df_rewards_smooth = df_rewards
    # compute mean and stds
    mean = df_rewards_smooth.mean(axis=1)
    std = df_rewards_smooth.std(axis=1)
                
    return mean, std
    