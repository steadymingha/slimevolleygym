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

# game settings:

RENDER_MODE = True


if __name__=="__main__":
    """
    Example of how to use Gym env, in single or multiplayer setting
    
    Humans can override controls:
    
    blue Agent:
    W - Jump
    A - Left
    D - Right
    
    Yellow Agent:
    Up Arrow, Left Arrow, Right Arrow
    """

    if RENDER_MODE:
        import pyglet
        from pyglet.window import key

        from time import sleep

    manualAction = [0, 0, 0] # forward, backward, jump
    otherManualAction = [0, 0, 0]
    manualMode = False
    otherManualMode = False

    # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py


    # policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player

    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))
    #env.seed(689)
    state = env.reset()
    if RENDER_MODE:
        env.render()

    # obs = env.reset()

    steps = 0
    total_reward = 0
    action = np.array([0, 0, 0])

    done = False
    ##################################################MH###########################

    env_name = "SlimeVolley-v0"

    has_continuous_action_space = False
    max_ep_len = 3000  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving

    render = True  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 10  # total num of testing episodes

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

    #####################################################

    # env = gym.make(env_name)
    # state = env.reset()
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
      action_dim = env.action_space.shape[0]
    else:
      action_dim = env.action_space.n

    # initialize a PPO agent

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                  action_std)
    # checkpoint_path = 'PPO_preTrained/SlimeVolley-v0/best/PPO_SlimeVolley-v0_0_02709.pth'
    checkpoint_path = 'PPO_preTrained/SlimeVolley-v0/best/PPO_SlimeVolley-v0_0_0.pth'
    print("loading network ..")

    ppo_agent.load(checkpoint_path)

    while not done:

        action = ppo_agent.select_action(state)

        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if RENDER_MODE:
          env.render()
          sleep(0.02) # 0.01

    env.close()
    print("cumulative score", total_reward)
