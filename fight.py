"""
State mode (Optional Human vs Built-in AI)

FPS (no-render): 100000 steps /7.956 seconds. 12.5K/s.
"""

import math
import numpy as np
import gym
import torch
# import gymnasium as gym
import slimevolleygym
from PPO import PPO
from gym.wrappers.monitoring.video_recorder import VideoRecorder



MH = False
SH_r = False
SH_l = False
Video_r = False
Video_l = False
YS_r = False
YS_l = False


# Video_l =
YS_r = True



Record = False
video_name = 'Skynet/record/좌영상우승현.mp4'







class PPOPolicy:
    def __init__(self, path):
        self.model = PPO.load(path)

    def select_action(self, obs):
        action, state = self.model.predict(obs, deterministic=True)
        return action

if __name__=="__main__":
    from pyglet.window import key
    from time import sleep

    env = gym.make("SlimeVolley-v0")
    env.seed(np.random.randint(0, 10000))

    env.render()
    if Record: video_recorder = VideoRecorder(env, video_name)


    # ##################################################MH###########################
    # # initialize a MH agent
    # ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    # checkpoint_path = '/home/user/RLstudy/slimevolleygym/PPO_preTrained/SlimeVolley-v0/PPO_SlimeVolley-v0_0_0.pth'
    ##################################################MH###########################
    # initialize a SH agent
    if SH_r:
        from collections import namedtuple
        from Skynet.seunghyun_slime_a2c_v2 import Policy
        gamma = 0.99
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        lr = 5e-4
        sh_agent = Policy(12,6)
        checkpoint_path = 'Skynet/fight_day2/volley_actor_v4_right.pth'
        sh_agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    elif SH_l:
        from collections import namedtuple
        from Skynet.seunghyun_slime_a2c_v2 import Policy

        gamma = 0.99
        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        lr = 5e-4
        sh_agent = Policy(12, 6)
        checkpoint_path = 'Skynet/fight_day2/volley_actor_v4_left.pth'
        sh_agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    #################################################MH###########################
    # # intialize a Video agent
    if Video_r:
        from stable_baselines3 import PPO
        video_agent = PPOPolicy('Skynet/fight_day2/best_model_video.zip')
    elif Video_l:
        from stable_baselines3 import PPO
        video_agent = PPOPolicy('Skynet/fight_day2/best_model_video.zip')

    ##################################################MH###########################
    # initialize a YS agent
    if YS_r:
        from Skynet.yongsun_final_script import ActorCritic
        ys_agent = ActorCritic(12, 128, 6)
        checkpoint_path = 'Skynet/fight_day2/yongsun_right.pth'
        ys_agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    elif YS_l:
        from Skynet.yongsun_final_script import ActorCritic
        ys_agent = ActorCritic(12, 128, 6)
        checkpoint_path = 'Skynet/fight_day2/yongsun_left.pth'
        ys_agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    #################################################MH###########################

    print("loading network ..")
    state_right = env.reset()
    state_left = state_right.copy()
    total_reward = 0
    done = False

    while not done:
        if SH_r:
            action_right = sh_agent.select_action(state_right)
            action_right = env.action_table[action_right]
        elif SH_l:
            action_left = sh_agent.select_action(state_left)
            action_left = env.action_table[action_left]

        if Video_r:
            action_right = video_agent.select_action(state_right)
        elif Video_l:
            action_left = video_agent.select_action(state_left)

        if YS_r:
            # action_left = ys_agent.select_action(state_left)
            action_right = ys_agent.select_action(state_right)
        elif YS_l:
            action_left = ys_agent.select_action(state_left)

        state_right, reward, done, info = env.step(action_right)#, action_left)
        state_left = info['otherState']

        total_reward += reward

        if Record: video_recorder.capture_frame()
        env.render()
        sleep(0.01)

    env.close()
    if Record: video_recorder.close()
    print("cumulative score", total_reward)
