import numpy as np
import torch
import os

class HashTable(object):
  def __init__(self, args):
    self.table = {}
    self.state_action_table = {}
    self.state_action_count = 0
    self.hash_dim = args.hash_dim
    self.device = args.device
    self.state_dim = 46 * args.history_length if args.env_type == 'sepsis' else args.history_length * 6 if args.env_type == 'hiv' else 84 * 84 * args.history_length if args.env_type == "atari" else args.history_length * args.state_dim
    if args.env_type != "gym":
      self.A = torch.randn([self.hash_dim, self.state_dim], device=args.device)
    else:
      self.A = torch.eye(args.state_dim, device=args.device)
      self.low = torch.from_numpy(args.observation_space.low).reshape(-1, 1).to(args.device)
      self.high = torch.from_numpy(args.observation_space.high).reshape(-1, 1).to(args.device)
      self.interval = (self.high - self.low) / self.hash_dim
      
    self.args = args
    self.info_matrix = {}
    self.previous_info_value = {}
    self.cur_info_index = None

  def step(self, state, action, count_action_state, need_info_matrix=False, info_index=0):
    hash_item, state_action_hash_item, hidden = self.hash_state(state, action)
    if hash_item in self.table:
      self.table[hash_item] += 1
    else:
      self.table[hash_item] = 1
    if need_info_matrix:
      self.step_info_matrix(hidden, action, info_index)

    if count_action_state:
      if state_action_hash_item in self.state_action_table:
        self.state_action_table[state_action_hash_item] += 1
      else:
        self.state_action_table[state_action_hash_item] = 1
      self.state_action_count = self.state_action_table[state_action_hash_item]
    return self.table[hash_item]

  def hash_state(self, state, action):
    hidden = self.A.matmul(state.contiguous().view(-1, 1))
    if self.args.env_type != "gym":
      state_hash = ''.join(map(lambda x: 'a' if x > 0 else 'b', hidden))
      state_action_hash = state_hash + str(action)
    else:
      state_hash = (hidden - self.low) / self.interval
      state_hash = '.'.join(map(lambda x: str(int(x.item())), state_hash))
      state_action_hash = state_hash + "." + str(action)
    return state_hash, state_action_hash, hidden
  
  def step_info_matrix(self, hidden, action, info_index):
    # hidden = self.A.matmul(state.contiguous().view(-1, 1))
    hidden_with_action = torch.cat([hidden, torch.tensor([[action]]).type_as(hidden)])
    cur_matrix = hidden_with_action @ hidden_with_action.T
    if self.info_matrix.get(info_index, None) is None:
      self.info_matrix[info_index] = cur_matrix
    else:
      self.info_matrix[info_index] += cur_matrix
    self.cur_info_index = info_index
  
  @property
  def info_matrix_value(self):
    (evals, _) = torch.eig(self.info_matrix[self.cur_info_index], eigenvectors=False)
    evals = evals[:, 0]
    cur_info_value = evals.abs().min().item()
    # cur_info_value = evals.abs().prod().item()
    # cur_info_value = torch.det(self.info_matrix[self.cur_info_index] + torch.eye(self.info_matrix[self.cur_info_index].size(0), device=self.device))
    # cur_info_value = torch.det(self.info_matrix[self.cur_info_index]).abs()
    previous_value = self.previous_info_value.get(self.cur_info_index, None)
    out_flag = previous_value is None or previous_value == 0 or cur_info_value / previous_value >= self.args.info_matrix_ratio
    if out_flag:
      self.previous_info_value[self.cur_info_index] = cur_info_value
    return cur_info_value, out_flag

  def save(self, path, name='hash.pth'):
    torch.save({'A': self.A, 'table': self.table, 'state_action_table': self.state_action_table}, os.path.join(path, name))

  def load(self, path, name='hash.pth'):
    state = torch.load(os.path.join(path, name), map_location='cpu')
    self.A = state['A'].to(device=self.device)
    self.table = state['table']
    self.state_action_table = state.get('state_action_table', {})
