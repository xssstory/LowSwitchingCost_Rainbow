# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
from tqdm import tqdm, trange
import numpy as np

import atari_py
from env import Env

import argparse
from agent import Agent

def get_args():
  parser = argparse.ArgumentParser(description='Rainbow')
  parser.add_argument('--model-dir', type=str, default='results', help='Experiment ID')
  parser.add_argument('--seed', type=int, default=1, help='Random seed')
  parser.add_argument('--game', default='battle_zone', choices=atari_py.list_games())
  parser.add_argument('--start-ckpt', default=100000, type=int)
  parser.add_argument('--env-type', default='atari', choices=['atari', 'sepsis', 'hiv', 'gym'])
  parser.add_argument('--deploy-policy', default=None, choices=['fixed', 'exp', 'dqn-feature', "reset_feature", 'q-value', 'dqn-feature-min',
                                                                'reset', 'policy', 'policy_adapt', 'policy_diverge', 'reset_policy', 'visited', "info-matrix",
                                                                'reset_feature_force', 'feature_lowf'])
  parser.add_argument("--info-matrix-interval", default=10, type=int)
  parser.add_argument("--info-matrix-ratio", default=2, type=float)
  parser.add_argument("--use-gradient-weight", default=False, action="store_true")
  parser.add_argument("--adaptive-softmax", default=False, action="store_true")
  parser.add_argument('--record-action-diff', default=False, action="store_true")
  parser.add_argument('--record-feature-sim', default=False, action="store_true")
  parser.add_argument('--switch-memory-priority', default=True, type=eval)
  parser.add_argument('--switch-bsz', default=32, type=int)
  parser.add_argument('--switch-sample-strategy', default=None, choices=['uniform', 'recent'], type=str, help="only useful when switch-memory-priority is False")
  parser.add_argument('--switch-memory-capcacity', default=1000000, type=int)
  parser.add_argument('--diverge-threshold', default=0.2, type=float)
  parser.add_argument('--delploy-interval', default=1, type=int)
  parser.add_argument('--force-interval', default=1000, type=int, help='For reset_feature_force strategy')
  parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
  parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
  parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
  parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
  parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
  parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
  parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
  parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
  parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
  parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
  parser.add_argument("--gradient-steps", default=1, type=int)
  parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
  parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
  parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
  parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
  parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
  # parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
  parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
  parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
  parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
  parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
  parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
  # parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
  # parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
  parser.add_argument('--evaluate-dir', default=None, type=str)
  parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
  parser.add_argument('--evaluation-episodes', type=int, default=1, metavar='N', help='Number of evaluation episodes to average over')
  # parser.add_argument('--state-visitation-episodes', type=int, default=10, help='Number of evaluation episodes to measure state visitation')
  # parser.add_argument('--result-dir', default='results/', type=str)
  # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
  # parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
  parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
  # parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
  # parser.add_argument('--checkpoint-interval', default=0, type=int, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
  # parser.add_argument('--memory', help='Path to save/load the memory from')
  parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

  return parser.parse_args()


# Test DQN

def test(args, T, dqn, metrics, env_class):
  env = env_class(args, training=False)
  metrics['steps'].append(T)

  T_rewards = []

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    state, reward_sum, done = env.reset(), 0, False
    for step in range(args.max_episode_length):     
      # if args.explore_eps is None:
      action = dqn.act(state)  # Choose an action greedily (with noisy weights)
      # else:
      #   action = dqn.act_e_greedy(state)
      state, reward, done, _ = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()
      if getattr(env, 'life_termination', None) and args.game == "breakout":
        env.ale.act(1)
        env.life_termination = False
      if done:
        T_rewards.append(reward_sum)
        break
  env.close()

  avg_reward = sum(T_rewards) / len(T_rewards)

  metrics['rewards'].append(T_rewards)

  assert len(metrics['steps']) == len(metrics['rewards']) == len(metrics['nums_deploy'])

  return avg_reward

if __name__ == "__main__":
  args = get_args()
  args.device = torch.device('cpu')
  args.env_type = 'atari'
  print(args.evaluation_interval)
  for game in sorted(atari_py.list_games()):
    print("*" * 50, game, "*" * 50)
    if not os.path.exists(os.path.join(args.model_dir, game)) or game == "montezuma_revenge":
      continue
    for method in ['none', 'fix_1000', 'feature0.98', 'feature0.99', 'info', 'visited', 'reset_feature_force', 'reset_feature_force10k', 'policy', "policy0.25"]:
      for seed in [1, 2, 3]:
        model_dir = f'{args.model_dir}/{game}/{method}.{seed}'

        if os.path.exists(os.path.join(model_dir, 'new_metrics_{}.pth'.format(args.evaluation_interval) if args.evaluation_interval != 100000 else 'new_metrics.pth' )):
          check_metrics = torch.load(os.path.join(model_dir, 'new_metrics_{}.pth'.format(args.evaluation_interval) if args.evaluation_interval != 100000 else 'new_metrics.pth'))
          if check_metrics['steps'][-1] >= 3e6:
            print('already exist', game, method, seed)
            continue
        args.game = game
        env = Env(args)
        dqn = Agent(args, env)
        try:
          old_metrics = torch.load(os.path.join(model_dir, 'metrics.pth'))
          if old_metrics['steps'][-1] <= 3e6:
            continue
        except:
          continue
        metrics = {'steps': [], 'rewards': [], 'nums_deploy': []}

        for ckpt in tqdm([20000] + list(range(args.start_ckpt, int(3.1e6), args.evaluation_interval)), ncols=50):
          idx = old_metrics['steps'].index(ckpt)
          metrics['nums_deploy'].append(old_metrics['nums_deploy'][idx])

          model_path = os.path.join(model_dir, f'checkpoint_{ckpt}.pth')
          try:
            dqn.load(model_path)
          except FileNotFoundError:
            continue
          test(args, ckpt, dqn, metrics, Env)

          tqdm.write('game : {}, method: {}, seed, {}, Step: {:.2f}, Rewards: {:.2f}, Num_deploy: {}'.format(game, method, seed, metrics['steps'][-1], np.mean(metrics['rewards'][-1]), metrics['nums_deploy'][-1]))
          torch.save(metrics, os.path.join(model_dir, 'new_metrics_{}.pth'.format(args.evaluation_interval) if args.evaluation_interval != 100000 else 'new_metrics.pth'))
