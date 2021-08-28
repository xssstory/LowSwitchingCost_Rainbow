# -*- coding: utf-8 -*-
from __future__ import division
from collections import namedtuple
import numpy as np
import torch
import time


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
# blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
    self.data = np.array([None] * size)  # Wrap-around cyclic buffer
    self.max = 1  # Initial max value to return (1 = 1^ω)

  # Propagates value up tree given a tree index
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate(parent, value)

  # Updates value given a tree index
  def update(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate(index, value)  # Propagate value
    self.max = max(value, self.max)

  def append(self, data, value):
    self.data[self.index] = data  # Store data in underlying data structure
    self.update(self.index + self.size - 1, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of a value in sum tree
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      return index
    elif value <= self.sum_tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.sum_tree[left])

  def find_parallel(self, value):
    if isinstance(value, float):
      value = np.array([value])
    # debug_value = value.copy()
    assert 0 <= np.min(value)
    assert np.max(value) <= self.sum_tree[0] + 1e-5
    assert isinstance(value[0], float)
    idx = np.zeros(len(value), dtype=int)
    cont = np.ones(len(value), dtype=bool)

    while np.any(cont):
      idx[cont] = 2 * idx[cont] + 1
      value_new = np.where(self.sum_tree[idx] <= value, value - self.sum_tree[idx], value)
      idx = np.where(np.logical_or(self.sum_tree[idx] > value, np.logical_not(cont)), idx, idx + 1)
      value = value_new
      cont = idx < self.size - 1
    index = idx
    data_index = index - self.size + 1
    # For debugging
    '''
    for i in range(debug_value.shape[0]):
        debug_result = self.find(debug_value[i])
        assert abs(debug_result[0] - self.sum_tree[index][i])<1e-5
        assert abs(debug_result[1] - data_index[i]) < 1e-5
        assert abs(debug_result[2] - index[i]) < 1e-5
    '''
    if len(data_index) == 1:
        index = index[0]
        data_index = data_index[0]
    # print(debug_result, self.sum_tree[index], data_index, index)
    return (self.sum_tree[index], data_index, index)

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]

class ReplayMemory():
  def __init__(self, args, capacity):
    if args.env_type == 'atari':
      self.blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)
      self.state_int = True
    elif args.env_type == 'sepsis':
      self.blank_trans = Transition(0, torch.zeros(46, 1, 1, dtype=torch.float32), None, 0, False)
      self.state_int = False
    elif args.env_type == 'hiv':
      self.blank_trans = Transition(0, torch.zeros(6, 1, dtype=torch.float32), None, 0, False)
      self.state_int = False
    else:
      self.blank_trans = Transition(0, torch.zeros(args.state_dim, 1, dtype=torch.float32), None, 0, False)
      self.state_int = False
    self.device = args.device
    self.capacity = capacity
    self.history = args.history_length
    self.discount = args.discount
    self.n = args.multi_step
    self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
    self.priority_exponent = args.priority_exponent
    self.t = 0  # Internal episode timestep counter
    self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

  # Adds state and action at time t, reward and terminal at time t + 1
  def append(self, state, action, reward, terminal):
    if self.state_int:
      state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
    else:
      state = state[-1].mul(255).to(dtype=torch.float32, device=torch.device('cpu'))
    self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)  # Store new transition with maximum priority
    self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

  # Returns a transition with blank states where appropriate
  def _get_transition(self, idx):
    transition = np.array([None] * (self.history + self.n))
    transition[self.history - 1] = self.transitions.get(idx)
    for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
      if transition[t + 1].timestep == 0:
        transition[t] = self.blank_trans  # If future frame has timestep 0
      else:
        transition[t] = self.transitions.get(idx - self.history + 1 + t)
    for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
      if transition[t - 1].nonterminal:
        transition[t] = self.transitions.get(idx - self.history + 1 + t)
      else:
        transition[t] = self.blank_trans  # If prev (next) frame is terminal
    return transition

  # Returns a valid sample from a segment
  def _get_sample_from_segment(self, segment, i):
    valid = False
    num_fail = 0
    while not valid:
      if num_fail < 10:
        sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
      else:
        p_total = self.transitions.total()
        sample = np.random.uniform(1, p_total) 
      prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
      # Resample if transition straddled current index or probablity 0
      if (self.transitions.index - idx) % self.capacity > self.n and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
        valid = True  # Note that conditions are valid but extra conservative around buffer index 0
      else:
        num_fail += 1

    # Retrieve all required transition data (from t - h to t + n)
    transition = self._get_transition(idx)
    # Create un-discretised state and nth next state
    state = torch.stack([trans.state for trans in transition[:self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
    next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
    # Discrete action to be used as index
    action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
    # Mask for non-terminal nth next states
    nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)

    return prob, idx, tree_idx, state, action, R, next_state, nonterminal
  
  def _get_sample_from_idx(self, idx):

    tree_idx = (idx % self.transitions.size) + self.transitions.size - 1
    # Retrieve all required transition data (from t - h to t + n)
    transition = self._get_transition(idx)
    # Create un-discretised state and nth next state
    state = torch.stack([trans.state for trans in transition[:self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
    next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
    # Discrete action to be used as index
    action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
    # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
    R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
    # Mask for non-terminal nth next states
    nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)

    return idx, tree_idx, state, action, R, next_state, nonterminal

  def sample(self, batch_size):
    p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
    segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
    batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
    states, next_states, = torch.stack(states), torch.stack(next_states)
    actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
    probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
    return tree_idxs, states, actions, returns, next_states, nonterminals, weights

  def sample2(self, batch_size):
    # import time
    # start_time = time.time()
    idxs, probs = self._sample_proportional(batch_size)
    # print("batch size", batch_size, "sample tree time", time.time() - start_time)
    p_total = self.transitions.total()
    probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
    capacity = self.capacity if self.transitions.full else self.transitions.index
    weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
    weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                           device=self.device)  # Normalise by max importance-sampling weight from batch
    # start_time = time.time()
    encoded_sample = self._encode_sample(idxs)
    # print("batch size", batch_size, "encode sample time", time.time() - start_time)
    # encoded_sample = self._sample_from_idxs(idxs)[:-1]
    return tuple(list(encoded_sample) + [weights,])

  # def _sample_proportional(self, batch_size):
  #   p_total = self.transitions.total()
  #   prefixs = np.random.uniform(size=batch_size) * p_total
  #   idxs = []
  #   probs = []
  #   for i in range(batch_size):
  #     prob, idx, _ = self.transitions.find(prefixs[i])
  #     idxs.append(idx)
  #     probs.append(prob)
  #   return idxs, probs

  def _sample_proportional(self, batch_size):
    idxs, probs = [], []
    p_total = self.transitions.total()
    segment = p_total / batch_size
    samples = np.random.uniform(0, segment, size=batch_size) + np.arange(batch_size) * segment
    probs, idxs, tree_idxs = self.transitions.find_parallel(samples)
    valid_idx = np.logical_and(np.logical_and((self.transitions.index - idxs) % self.capacity > self.n, 
        (idxs - self.transitions.index) % self.capacity >= self.history), probs != 0)
    # print(idxs[:10], self.transitions.index, self.capacity, self.n, self.history)
    # print('original idx number', len(idxs), idxs)
    # print('valid idx number', np.sum(valid_idx), valid_idx.shape)
    probs = probs[valid_idx]
    idxs = idxs[valid_idx]
    if np.sum(valid_idx) < batch_size:
        additional_idx = np.random.choice(np.arange(len(idxs)), batch_size - len(idxs))
        probs = np.concatenate([probs, probs[additional_idx]])
        idxs = np.concatenate([idxs, idxs[additional_idx]])
    # print("valid idx", idxs, "number", idxs.shape)
    assert idxs.shape[0] == batch_size
    assert probs.shape[0] == batch_size
    '''
    for i in range(batch_size):
      valid = False
      num_fail = 0
      while not valid:
        if num_fail < 10:
          sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
        else:
          sample = np.random.uniform(1, p_total)
        prob, idx, tree_idx = self.transitions.find_parallel(sample)  # Retrieve sample from tree with un-normalised probability
        # Resample if transition straddled current index or probablity 0
        if (self.transitions.index - idx) % self.capacity > self.n and (
                idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
          valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        else:
          num_fail += 1
      idxs.append(idx)
      probs.append(prob)
    '''
    return idxs, probs

  def _encode_sample(self, idxs):
    states_buf, actions_buf, returns_buf, next_states_buf, nonterminals_buf = [], [], [], [], []
    states_buf = torch.zeros(len(idxs), self.history, *self.transitions.get(0).state.shape).float().to(self.device)
    next_states_buf = torch.zeros(len(idxs), self.history, *self.transitions.get(0).state.shape).float().to(self.device)
    tree_idxs = ((np.asarray(idxs) % self.transitions.size) + self.transitions.size - 1).tolist()
    duration = 0
    duration_stack = 0
    duration_create = 0
    for i, idx in enumerate(idxs):
      # start_time = time.time()
      transition = self._get_transition(idx).tolist()
      # duration += time.time() - start_time
      _, states, actions, rewards, nonterminals = zip(*transition)
      # start_time = time.time()
      _state = torch.stack(states[:self.history]).float().to(self.device) / 255
      _next_state = torch.stack(states[self.n: self.n + self.history]).float().to(self.device) / 255
      states_buf[i] = _state
      next_states_buf[i] = _next_state
      # duration_stack += time.time() - start_time
      _action = actions[self.history - 1]
      _reward = np.sum(np.power(self.discount, np.arange(self.n)) * np.asarray(rewards[self.history: self.history + self.n]))
      # start_time = time.time()
      _nonterminal = torch.tensor([nonterminals[self.history + self.n - 1]]).float()
      # duration_create = time.time() - start_time
      # states_buf.append(_state)
      actions_buf.append(_action)
      returns_buf.append(_reward)
      # next_states_buf.append(_next_state)
      nonterminals_buf.append(_nonterminal)
    # start_time = time.time()
    # states_buf = torch.stack(states_buf).to(self.device)
    # duration_stack += time.time() - start_time
    # start_time = time.time()
    actions_buf = torch.from_numpy(np.asarray(actions_buf).astype(np.int)).to(self.device)
    returns_buf = torch.from_numpy(np.asarray(returns_buf)).float().to(self.device)
    # duration_create += time.time() - start_time
    # start_time = time.time()
    # next_states_buf = torch.stack(next_states_buf).to(self.device)
    # duration_stack += time.time() - start_time
    nonterminals_buf = torch.stack(nonterminals_buf).to(self.device)
    # print("get transition time", duration, "stack time", duration_stack, "create time", duration_create)
    # print(states_buf.shape, actions_buf.shape, returns_buf.shape, next_states_buf.shape, nonterminals_buf.shape)
    return tree_idxs, states_buf, actions_buf, returns_buf, next_states_buf, nonterminals_buf

  def _sample_from_idxs(self, idxs):
    batch = [self._get_sample_from_idx(i) for i in idxs]  # Get batch of valid samples
    idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
    states, next_states, = torch.stack(states), torch.stack(next_states)
    actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)

    return tree_idxs, states, actions, returns, next_states, nonterminals, None
  
  def uniform_sample(self, batch_size):
    idxs = np.random.choice(self.transitions.size, batch_size) if self.transitions.full else np.random.choice(self.transitions.index - 1, batch_size) 
    return self._sample_from_idxs(idxs)

  def sample_recent(self, batch_size):
    latest_idx = self.transitions.index - self.n
    latest_idx = latest_idx + self.transitions.size if latest_idx - batch_size < 0 else latest_idx
    idxs = np.arange(latest_idx - batch_size, latest_idx)
    return self._sample_from_idxs(idxs)
  
  def uniform_sample_from_recent(self, batch_size, sample_length):
    assert sample_length <= self.transitions.size 
    latest_idx = self.transitions.index - self.n
    if self.transitions.data[self.transitions.index] is None:
      low = max(self.n, self.transitions.index - sample_length)
      idxs = np.random.randint(low=low, high=latest_idx, size=batch_size)
    else:
      latest_idx = latest_idx + self.transitions.size if latest_idx - sample_length < 0 else latest_idx
      idxs = np.random.randint(low=latest_idx - sample_length, high=latest_idx, size=batch_size)
    return self._sample_from_idxs(idxs)

  def update_priorities(self, idxs, priorities):
    priorities = np.power(priorities, self.priority_exponent)
    [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == self.capacity:
      raise StopIteration
    # Create stack of states
    state_stack = [None] * self.history
    state_stack[-1] = self.transitions.data[self.current_idx].state
    prev_timestep = self.transitions.data[self.current_idx].timestep
    for t in reversed(range(self.history - 1)):
      if prev_timestep == 0:
        state_stack[t] = self.blank_trans.state  # If future frame has timestep 0
      else:
        state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
        prev_timestep -= 1
    state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
    self.current_idx += 1
    return state

  next = __next__  # Alias __next__ for Python 2 compatibility
