"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
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
    from pyglet.window import key
    from time import sleep

  env = gym.make( "SlimeVolley-v0")
  # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

  policy = slimevolleygym.BaselinePolicy() # defaults to use RNN Baseline for player

  state_dim = env.observation_space.shape[0]
  num_node = 64
  action_dim = 6#env.action_space.n
  K_epochs = 10  # 80  # update policy for K epochs in one PPO update
  eps_clip = 0.2  # clip parameter for PPO
  gamma = 0.99  # discount factor
  lr_actor = 0.0003  # learning rate for actor network
  lr_critic = 0.001  # 0.001  # learning rate for critic network

  ppo_agent = PPO(state_dim, num_node, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

  env.seed(np.random.randint(0, 10000))
  #env.seed(689)

  obs = env.reset()

  if RENDER_MODE:
    env.render()

  # obs = env.reset()

  steps = 0
  total_reward = 0
  action = np.array([0, 0, 0])

  done = False
  checkpoint_path = 'PPO_preTrained/SlimeVolley-v0/SlimeVolley-v0_0_0_selfplay.pth'
  print("loading network ..")

  ppo_agent.load(checkpoint_path)
  while not done:

    action = ppo_agent.select_action(obs)
    action = env.action_table[action]
    # action = policy.act(obs)
    obs, reward, done, _ = env.step(action)

    if done:
      # break
      env.reset()

    total_reward += reward

    if RENDER_MODE:
      env.render()

  env.close()
  print("cumulative score", total_reward)