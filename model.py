# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = nn.Linear(self.conv_output_size, args.hidden_size)
    self.fc_h_a = nn.Linear(self.conv_output_size, args.hidden_size)
    self.fc_z_v = nn.Linear(args.hidden_size, self.atoms)
    self.fc_z_a = nn.Linear(args.hidden_size, action_space * self.atoms)

  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    pass
  
  def extract(self, x):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    return x
  
  def feature2Q(self, x, log=False):
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

class SepsisDqn(nn.Module):
  def __init__(self, args, action_space):
    super(SepsisDqn, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    self.input_size = args.history_length * 46 if args.env_type == 'sepsis' else args.history_length * 6 if args.env_type == 'hiv' else None

    self.fc_forward = nn.Linear(self.input_size, args.hidden_size)
    self.fc_forward_2 = nn.Linear(args.hidden_size, args.hidden_size)

    self.fc_h_v = nn.Linear(args.hidden_size, args.hidden_size)
    self.fc_h_a = nn.Linear(args.hidden_size, args.hidden_size)
    self.fc_z_v = nn.Linear(args.hidden_size, self.atoms)
    self.fc_z_a = nn.Linear(args.hidden_size, action_space * self.atoms)

  def forward(self, x, log=False):
 
    x = x.view(-1, self.input_size)
    x = self.fc_forward(x)
    x = F.relu(x)
    x = self.fc_forward_2(x)
    x = F.relu(x)

    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    pass
  
  def extract(self, x):
    x = x.view(-1, self.input_size)
    x = self.fc_forward(x)
    x = F.relu(x)
    x = self.fc_forward_2(x)
    x = F.relu(x)
    return x

class GymDqn(nn.Module):
  def __init__(self, args, action_space):
    super(GymDqn, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    self.input_size = args.history_length * args.state_dim

    self.fc_forward = nn.Linear(self.input_size, args.hidden_size)
    # self.fc_forward_2 = nn.Linear(args.hidden_size, args.hidden_size)

    self.fc_h_v = nn.Linear(args.hidden_size, args.hidden_size)
    self.fc_h_a = nn.Linear(args.hidden_size, args.hidden_size)
    self.fc_z_v = nn.Linear(args.hidden_size, action_space)
    self.fc_z_a = nn.Linear(args.hidden_size, action_space)
    # self.fc_z_v = nn.Linear(args.hidden_size, self.atoms)
    # self.fc_z_a = nn.Linear(args.hidden_size, action_space * self.atoms)

  def forward(self, x, log=False):
 
    # x = x.view(-1, self.input_size)
    # x = self.fc_forward(x)
    # x = F.relu(x)
    # x = self.fc_forward_2(x)
    # x = F.relu(x)
    x = self.extract(x)

    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    # v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    # if log:  # Use log softmax for numerical stability
      # q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    # else:
      # q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    pass
  
  def extract(self, x):
    x = x.view(-1, self.input_size)
    x = self.fc_forward(x)
    x = F.relu(x)
    # x = self.fc_forward_2(x)
    # x = F.relu(x)
    return x