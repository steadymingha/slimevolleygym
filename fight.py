"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
# import gymnasium as gym
import slimevolleygym
from PPO import PPO

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

RENDER_MODE = True


if __name__=="__main__":


    if RENDER_MODE:
        import pyglet
        from pyglet.window import key
        from time import sleep

    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))

    state = env.reset()
    if RENDER_MODE:
        env.render()





    ##################################################MH###########################
    # initialize a MH agent
    state_dim = env.observation_space.shape[0]
    action_dim = 6  # env.action_space.n
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    K_epochs = 80  # update policy for K epochs

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    checkpoint_path = '/home/user/RLstudy/slimevolleygym/PPO_preTrained/SlimeVolley-v0/PPO_SlimeVolley-v0_0_0.pth'
    print("loading network ..")
    ppo_agent.load(checkpoint_path)
    #################################################MH###########################


    total_reward = 0
    done = False
    while not done:

        action = ppo_agent.select_action(state,1)
        # action = env.action_table[action]
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if RENDER_MODE:
          env.render()
          sleep(0.02)

        print(total_reward)

    env.close()
    print("cumulative score", total_reward)
