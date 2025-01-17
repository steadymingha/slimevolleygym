import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] ="ea4767cdbc1f73ed2efc7cf70d83a1526fea8042"
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool
import slimevolleygym

from PPO_selfplay import PPO
from time import sleep
import wandb

####################################################################################
id = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb.init(project="SlimeVolley-v0", name =id)
print("This run's ID:", wandb.run.id)
wandb.run.save()
# ################################### Training ###################################
def train():

    ####### initialize environment hyperp
    env_name = "SlimeVolley-v0"

    max_ep_len = 3000  # max timesteps in one episode


    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 10#80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001 #0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = gym.make(env_name)
    env.seed(np.random.randint(0, 10000))
    # state space dimension
    state_dim = env.observation_space.shape[0]


    action_dim = 6  #env.action_space.n


    ################### checkpointing ###################
    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "{}_{}_{}_selfplay.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, 64, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # printing and logging variables

    time_step = 0
    i_episode = 0
######################################################################
    pre_rewards = 0
    # training loop
    while True:

        state_right = env.reset()
        state_left = state_right
        episode_rewards = 0

        for t in range(max_ep_len + 1):
            # select action with policy
            action_right = ppo_agent.select_action(state_right,1)
            action_right = env.action_table[action_right]

            action_left = ppo_agent.select_action(state_left)
            action_left = env.action_table[action_left]

            state_right, reward, done, info = env.step(action_right, action_left)
            state_left = info['otherState']

            # saving reward and is_terminals for right
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)



            time_step += 1
            episode_rewards += reward


            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            if done:
                break

        wandb.log({"Episode Reward": episode_rewards})

        # if i_episode % 10 == 0:
        #     print("Episode: {}, reward: {}".format(i_episode, episode_rewards))
        if pre_rewards < episode_rewards:
            torch.save(ppo_agent.policy.state_dict(), checkpoint_path)

        i_episode += 1
        pre_rewards = episode_rewards

    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()







