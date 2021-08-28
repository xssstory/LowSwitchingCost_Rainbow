# -*- coding: utf-8 -*-
from collections import deque
import random
import numpy as np
import atari_py
import cv2
import torch

class Env():
  def __init__(self, args, training=True):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    seed = args.seed if training else args.seed + 1 # different seeds for training and evaluation
    self.ale.setInt('random_seed', seed)
    if training:
      self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    else:
      self.ale.setInt('max_num_frames_per_episode', 108e3)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    self.game = args.game
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = training  # Consistent with model training mode

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
      if self.game == "breakout":
        self.ale.act(1)
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for i in range(np.random.randint(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
      if self.game == "breakout":
        self.ale.act(1)
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training or self.game == "breakout":
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
        if not self.training:
          done = False
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done, {}

  # # Uses loss of life as terminal signal
  # def train(self):
  #   self.training = True

  # # Uses standard terminal signal
  # def eval(self):
  #   self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()

# import whynot.gym as gym
from whynot.gym.envs import registry
class WhyNotEnv():

  def __init__(self, args, training=True):
    # self.env = gym.make('HIV-v0')
    self.env = registry.spec('HIV-v0').entry_point()
    seed = args.seed if training else args.seed + 1 # different seeds for training and evaluation
    self.env.seed(seed)
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.device = args.device

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(6, 1, device=self.device))

  def reset(self):
    s = self.env.reset()
    self._reset_buffer()
    self.state_buffer.append(torch.tensor(s.reshape([-1, 1]), device=self.device, dtype=torch.float32))
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    s, reward, done, info = self.env.step(action)
    reward /= 1e5
    self.state_buffer.append(torch.tensor(s.reshape([-1, 1]), device=self.device, dtype=torch.float32))
    return torch.stack(list(self.state_buffer), 0), reward, done, info
  
  def action_space(self):
    return self.env.action_space.n
  
  def render(self):
    return self.env.render()
  
  def close(self):
    self.env.close()
    del self.env
  

import gym

class GymEnv():

  def __init__(self, args, training=True):
    self.env = gym.make(args.game)
    seed = args.seed if training else args.seed + 1 # different seeds for training and evaluation
    self.env.seed(seed)
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.device = args.device
    self.state_dim = self.env.observation_space.shape[0]

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(self.state_dim, 1, device=self.device))

  def reset(self):
    s = self.env.reset()
    self._reset_buffer()
    self.state_buffer.append(torch.tensor(s.reshape([-1, 1]), device=self.device, dtype=torch.float32))
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    s, reward, done, info = self.env.step(action)
    self.state_buffer.append(torch.tensor(s.reshape([-1, 1]), device=self.device, dtype=torch.float32))
    return torch.stack(list(self.state_buffer), 0), reward, done, info
  
  def action_space(self):
    return self.env.action_space.n
  
  def render(self):
    return self.env.render()
  
  def close(self):
    self.env.close()
    del self.env
