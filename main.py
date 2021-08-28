# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os, shutil
import pickle

import atari_py
import numpy as np
import torch
from tqdm import trange, tqdm

from agent import Agent
from env import Env, WhyNotEnv, GymEnv
from memory import ReplayMemory
from test import test, eval_visitation

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'gym-sepsis'))
from gym_sepsis.envs.sepsis_env import SepsisEnv

from hash_count import HashTable
import math

ENV_DIC = {
  'atari': Env,
  'sepsis': SepsisEnv,
  'hiv': WhyNotEnv,
  "gym": GymEnv,
}

def min_interval_type(s):
  try:
    value = int(s)
  except ValueError:
    if s.startswith('adaptive.'):
      value = s
    else: 
      raise argparse.ArgumentTypeError('min-interval must be a integer or "adaptive.int"')
  return value


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
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
parser.add_argument('--min-interval', default=0, type=min_interval_type, help='This setting is useful for dqn-feature-min')
parser.add_argument('--exp-base', default=2, type=float, help='This setting is useful for exp')
parser.add_argument('--feature-threshold', default=0.98, type=float, help='This setting is useful for dqn-feature and dqn-feature-fixed')
parser.add_argument('--td-error-threshold', default=0.1, type=float, help='This settiong is useful for td-error')
parser.add_argument('--q-value-threshold', default=0.05, type=float, help='This setting is useful for q-value')
parser.add_argument('--policy-diff-threshold', default=0.05, type=float, help='This setting is useful for policy')
parser.add_argument("--explore-eps", default=None, type=eval)
parser.add_argument('--count-base-bonus', default=-1, type=float)
parser.add_argument('--hash-dim', default=32, type=int)
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games() + ["MountainCar-v0", "Acrobot-v1"], help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
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
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluate-dir', default=None, type=str)
parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--state-visitation-episodes', type=int, default=10, help='Number of evaluation episodes to measure state visitation')
parser.add_argument('--result-dir', default='results/', type=str)
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, type=int, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join(args.result_dir, args.id)
if os.path.exists(results_dir):
  print("Warning: %s already exists!" % results_dir)
  r = input("Do you want to remove %s? [Y] " % results_dir)
  if r == "Y":
    shutil.rmtree(results_dir)
  else:
    print("Keep the directory and exit.")
    exit()
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

with open(os.path.join(results_dir, 'params.txt'), 'w') as f:
  for k, v in vars(args).items():
    f.write(' ' * 26 + k + ': ' + str(v) + '\n')

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf'), 'nums_deploy': [],
           'episode_length': [], 'episode_reward': [], 'episode_step': [], 'action_diff': [], "feature_sim": []}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
  torch.backends.cudnn.enabled = args.enable_cudnn
else:
  args.device = torch.device('cpu')

print(args.device)
# Simple ISO 8601 timestamped logger
def log(s):
  # print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
  tqdm.write('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)

# Shorten episode if deploy policy is reset
if args.deploy_policy == "reset_policy" or args.deploy_policy == "reset_feature":
  args.max_episode_length = 10000

# Environment
env = ENV_DIC[args.env_type](args)
action_space = env.action_space()
args.state_dim = getattr(env, "state_dim", None)
if isinstance(env, GymEnv):
  args.observation_space = env.env.observation_space

# if args.count_base_bonus > 0:
hash_table = HashTable(args)

# Agent
dqn = Agent(args, env)
if args.model and args.count_base_bonus > 0:
  hash_table.load(os.path.dirname(args.model))

# If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
  if done:
    state, done = env.reset(), False

  next_state, _, done, _ = env.step(np.random.randint(0, action_space))
  val_mem.append(state, None, None, done)
  state = next_state
  T += 1

# if args.evaluate_dir:
#   eval_metrics = {}
#   eval_metrics['steps'] = []
#   eval_metrics['reward'] = []
#   eval_hash_table = HashTable(args)
#   args.state_visitation_episodes = args.evaluation_episodes
#   import glob
#   files = glob.glob(args.evaluate_dir + "/*")
#   checkpoint_dic = {}
#   for f in files:
#     if "checkpoint" in f:
#       num = int(f[f.find('checkpoint_') + 11:].strip('.pth'))
#       checkpoint_dic[num] = f
#   state_visitation, total_steps, avg_reward, std_reward = eval_visitation(args, dqn, eval_hash_table, ENV_DIC[args.env_type])
#   print(avg_reward)
#   eval_metrics['steps'].append(0)
#   eval_metrics['reward'].append(avg_reward)
#   for k, v in sorted(checkpoint_dic.items(), key=lambda x: x[0]):
#     if k > 1.5e6:
#       break
#     dqn.load(v)
#     dqn.eval()
#     state_visitation, total_steps, avg_reward, std_reward = eval_visitation(args, dqn, eval_hash_table, ENV_DIC[args.env_type])
#     print(avg_reward)
#     eval_metrics['steps'].append(k)
#     eval_metrics['reward'].append(avg_reward)
#     torch.save(eval_metrics, os.path.join(results_dir, "eval_metrics.pth"))
#   import sys
#   sys.exit()

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  # avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, ENV_DIC[args.env_type], evaluate=True)  # Test
  # print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
  # eval_hash_table = HashTable(args)
  eval_hash_table = hash_table
  eval_hash_table.table = {}
  state_visitation, total_steps, avg_reward, std_reward = eval_visitation(args, dqn, eval_hash_table, ENV_DIC[args.env_type])
  print("Avg. reward:", avg_reward, "Std.:", std_reward, "Total steps", total_steps)
  # TODO: measure entropy? number of visited states?
  print("Number of visited states", len(state_visitation))
  visit_freq = np.asarray(list(state_visitation.values())) / total_steps
  # print("Visitation frequency", visit_freq)
  state_dist_entropy = np.sum(-visit_freq * np.log(visit_freq))
  print("Entropy", state_dist_entropy)
else:
  # Training loop
  dqn.train()
  T, done = 0, True
  episode_length, episode_reward = 0, 0

  visited_deploy_flag = False

  T_total = 3100000

  # for T in trange(1, args.T_max + 1):
  pbar = tqdm(total=T_total, ncols=50)
  for T in range(1, args.T_max + 1):
    if T > T_total:
      print(f"Terminate after {T_total} steps!")
      break
    if done:
      state, done = env.reset(), False
      metrics['episode_length'].append(episode_length)
      metrics['episode_reward'].append(episode_reward)
      metrics['episode_step'].append(T)
      episode_length, episode_reward = 0, 0
      last_done_T = T

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    if args.explore_eps is None:
      action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    else:
      init_eps = 1
      decay_step = args.explore_eps[0] if args.explore_eps[0] > 1 else args.explore_eps[0] * args.T_max
      final_eps = args.explore_eps[1]
      eps = max(init_eps - (init_eps - final_eps) / decay_step * T, final_eps)
      action = dqn.act_e_greedy(state, eps)
    next_state, reward, done, _ = env.step(action)  # Step
    episode_reward += reward
    episode_length += 1
    if args.count_base_bonus > 0:
      if args.deploy_policy == "info-matrix":
        info_index = (T - last_done_T) // args.info_matrix_interval
        reward = reward + args.count_base_bonus / math.sqrt(hash_table.step(state, action, T > args.learn_start and not visited_deploy_flag, True, info_index))
      else:
        reward = reward + args.count_base_bonus / math.sqrt(hash_table.step(state, action, T > args.learn_start and not visited_deploy_flag))

    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards      

    mem.append(state, action, reward, done)  # Append transition to memory

    # Train and test
    if T >= args.learn_start:
      # if (T % args.evaluation_interval == 0 and args.deploy_policy is None) or (not metrics['nums_deploy']) or (dqn.num_deploy > metrics['nums_deploy'][-1] and args.deploy_policy is not None): 
      #   if (not metrics['steps']) or T - metrics['steps'][-1] >= args.evaluation_interval:
      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir, ENV_DIC[args.env_type])  # Test

        dqn.save(results_dir, 'checkpoint_{}.pth'.format(T))
        try:
          hash_table.save(results_dir, 'hash.pth')
        except:
          pass

        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. last reward: %.3f' % np.mean(metrics['episode_reward'][-10:]) + ' last eps. length: %.3f' % np.mean(metrics['episode_length'][-10:])  + ' | Deploy: ' + str(metrics['nums_deploy'][-1]))
        dqn.train()  # Set DQN (online network) back to training mode

      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if args.deploy_policy == "reset":
        if T % args.replay_frequency == 0:
          dqn.learn(mem)
          if args.memory is not None:
            save_memory(mem, args.memory, args.disable_bzip_memory)
        # For reset deploy, it may happen when T mod replay-frequency != 0
        dqn.update_deploy_net(None, args, mem, is_reset=(T > 0 and done))
      elif args.deploy_policy in ["reset_policy", "reset_feature"]:
        if T % args.replay_frequency == 0:
          dqn.learn(mem)
          if args.memory is not None:
            save_memory(mem, args.memory, args.disable_bzip_memory)
        if done:
          dqn.update_deploy_net(None, args, mem)
      elif args.deploy_policy == "reset_feature_force":
        if T % args.replay_frequency == 0:
          dqn.learn(mem)
        if args.memory is not None:
          save_memory(mem, args.memory, args.disable_bzip_memory)
        dqn.update_deploy_net(T, args, mem, is_reset=(T > 0 and done))
      elif args.deploy_policy == "visited":
        count = hash_table.state_action_count
        if count <= 0 or count & count - 1 == 0:
          visited_deploy_flag = True
        if T % args.replay_frequency == 0:
          dqn.learn(mem)
          if visited_deploy_flag:
            dqn.update_deploy_net(None, args, mem)
            visited_deploy_flag = False
      elif args.deploy_policy == "info-matrix":
        if (T - last_done_T) % args.info_matrix_interval == args.info_matrix_interval - 1 or done:
          info_value, visited_deploy_flag = hash_table.info_matrix_value
        if T % args.replay_frequency == 0:
          dqn.learn(mem)
          if visited_deploy_flag:
            dqn.update_deploy_net(None, args, mem)
            visited_deploy_flag = False

      else:
        # if args.deploy_policy == "fixed":
        #     dqn.update_deploy_net(T, args, mem)
        if T % args.replay_frequency == 0:
          dqn.learn(mem)  # Train with n-step distributional double-Q learning
          if args.deploy_policy == "fixed":
            dqn.update_deploy_net(T, args, None)
          elif (args.deploy_policy in ["policy", "policy_diverge", "dqn-feature"]):
              if T % (args.replay_frequency * 8) == 0:
                  dqn.update_deploy_net(T, args, mem)
          elif args.deploy_policy == "feature_lowf":
              if T % (args.replay_frequency * 64) == 0:
                  dqn.update_deploy_net(T, args, mem)
          # elif args.deploy_policy != "fixed":
          else:
              dqn.update_deploy_net(T, args, mem)
          
          # If memory path provided, save it
          if args.memory is not None:
            save_memory(mem, args.memory, args.disable_bzip_memory)

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

      # Checkpoint the network
      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        dqn.save(results_dir, 'checkpoint_{}.pth'.format(T))
        try:
          hash_table.save(results_dir, 'hash.pth')
        except:
          pass

    state = next_state
    pbar.update(1)

env.close()
