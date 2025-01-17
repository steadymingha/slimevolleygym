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

from gym.wrappers.monitoring.video_recorder import VideoRecorder



MH_r = False
MH_l = False
SH_r = False
SH_l = False
Video_r = False
Video_l = False
YS_r = False
YS_l = False

MH_l = True
YS_r = True



Record = True
video_name = 'Skynet/ys_gujilgujil/좌명화우용선2_new.mp4'







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
    env.metadata['video.frames_per_second'] = 60
    env.render()
    if Record: video_recorder = VideoRecorder(env, video_name)


    # ##################################################MH###########################
    # initialize a MH agent
    from PPO_selfplay import PPO
    if MH_r or MH_l:
        mh_agent = PPO(12, 64, 6)
        checkpoint_path = 'PPO_preTrained/SlimeVolley-v0/SlimeVolley-v0_0_0_selfplay.pth'
        mh_agent.load(checkpoint_path)
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
        from Skynet.YS_V2 import ActorCritic
        ys_agent = ActorCritic(12, 128, 6)
        checkpoint_path = 'Skynet/fight_day2/yongsun-v2.pth'
        ys_agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    elif YS_l:
        from Skynet.YS_V2 import ActorCritic
        ys_agent = ActorCritic(12, 128, 6)
        checkpoint_path = 'Skynet/ys_gujilgujil/checkpoint.pth'
        ys_agent.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    #################################################MH###########################

    print("loading network ..")
    state_right = env.reset()
    state_left = state_right.copy()

    total_reward = 0
    done = False

    while not done:
        if MH_r:
            action_right = mh_agent.select_action(state_right)
            action_right = env.action_table[action_right]
        elif MH_l:
            action_left = mh_agent.select_action(state_left)
            action_left = env.action_table[action_left]

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
            # action_right = ys_agent.select_action(state_right)
            action = ys_agent.forward_predict(state_right)
            action_right = env.action_table[action]
        elif YS_l:
            action_left = ys_agent.forward_predict(state_left)
            action_left = env.action_table[action_left]

        state_right, reward, done, info = env.step(action_right, action_left)
        state_left = info['otherState']

        total_reward += reward

        if Record: video_recorder.capture_frame()
        env.render()
        # sleep(0.01)

    env.close()
    if Record: video_recorder.close()
    print("cumulative score", total_reward)
