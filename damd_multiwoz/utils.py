import logging
import json
import numpy as np
from collections import OrderedDict
import ontology
import pdb
from sklearn.model_selection import train_test_split

def py2np(list):
    return np.array(list)


def write_dict(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, indent=2)

def f1_score(label_list, pred_list):
    tp = len([t for t in pred_list if t in label_list])
    fp = max(0, len(pred_list) - tp)
    fn = max(0, len(label_list) - tp)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1

class Vocab(object):
    def __init__(self, vocab_size=0):
        self.vocab_size = vocab_size
        self.vocab_size_oov = 0   # get after construction
        self._idx2word = {}   #word + oov
        self._word2idx = {}   # word
        self._freq_dict = {}   #word + oov
        for w in ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>','<eos_u>', '<eos_r>',
                      '<eos_b>', '<eos_a>', '<go_d>','<eos_d>']:
            self._absolute_add_word(w)

    def _absolute_add_word(self, w):
        idx = len(self._idx2word)
        self._idx2word[idx] = w
        self._word2idx[w] = idx

    def add_word(self, word):
        if word not in self._freq_dict:
            self._freq_dict[word] = 0
        self._freq_dict[word] += 1

    def has_word(self, word):
        return self._freq_dict.get(word)

    def _add_to_vocab(self, word):
        if word not in self._word2idx:
            idx = len(self._idx2word)
            self._idx2word[idx] = word
            self._word2idx[word] = idx

    def construct(self):
        l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
        print('Vocabulary size including oov: %d' % (len(l) + len(self._idx2word)))
        if len(l) + len(self._idx2word) < self.vocab_size:
            logging.warning('actual label set smaller than that configured: {}/{}'
                            .format(len(l) + len(self._idx2word), self.vocab_size))
        for word in ontology.all_domains + ['general']:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_acts:
            word = '[' + word + ']'
            self._add_to_vocab(word)
        for word in ontology.all_slots:
            self._add_to_vocab(word)
        for word in l:
            if word.startswith('[value_') and word.endswith(']'):
                self._add_to_vocab(word)
        for word in l:
            self._add_to_vocab(word)
        self.vocab_size_oov = len(self._idx2word)

    def load_vocab(self, vocab_path):
        self._freq_dict = json.loads(open(vocab_path+'.freq.json', 'r').read())
        self._word2idx = json.loads(open(vocab_path+'.word2idx.json', 'r').read())
        self._idx2word = {}
        for w, idx in self._word2idx.items():
            self._idx2word[idx] = w
        self.vocab_size_oov = len(self._idx2word)
        print('vocab file loaded from "'+vocab_path+'"')
        print('Vocabulary size including oov: %d' % (self.vocab_size_oov))

    def save_vocab(self, vocab_path):
        _freq_dict = OrderedDict(sorted(self._freq_dict.items(), key=lambda kv:kv[1], reverse=True))
        write_dict(vocab_path+'.word2idx.json', self._word2idx)
        write_dict(vocab_path+'.freq.json', _freq_dict)


    def encode(self, word, include_oov=True):
        if include_oov:
            if self._word2idx.get(word, None) is None:
                self._add_to_vocab(word)
                self.vocab_size_oov = len(self._idx2word)
                #logging.info(ValueError('Unknown word: %s. Vocabulary should include oovs here.'%word))
                #raise ValueError('Unknown word: %s. Vocabulary should include oovs here.'%word)
            return self._word2idx[word]
        else:
            word = '<unk>' if word not in self._word2idx else word
            return self._word2idx[word]

    def sentence_encode(self, word_list):
        return [self.encode(_) for _ in word_list]

    def oov_idx_map(self, idx):
        return 2 if idx > self.vocab_size else idx

    def sentence_oov_map(self, index_list):
        return [self.oov_idx_map(_) for _ in index_list]


    def decode(self, idx, indicate_oov=False):
        if not self._idx2word.get(idx):
            raise ValueError('Error idx: %d. Vocabulary should include oovs here.'%idx)
        if not indicate_oov or idx<self.vocab_size:
            return self._idx2word[idx]
        else:
            return self._idx2word[idx]+'(o)'

    def sentence_decode(self, index_list, eos=None, indicate_oov=False):
        l = [self.decode(_, indicate_oov) for _ in index_list] # ['hi', '.', 'can', 'you', 'help', 'me', 'find', 'an', 'east', 'hotel', '?', '<eos_u>']
        if not eos or eos not in l:
            return ' '.join(l)
        else:
            idx = l.index(eos)
            return ' '.join(l[:idx]) # 'hi . can you... east hotel ?'

    def nl_decode(self, l, eos=None):
        return [self.sentence_decode(_, eos) + '\n' for _ in l]



def padSeqs(sequences, maxlen=None, truncated = False, pad_method='post',
                     trunc_method='pre', dtype='int32', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    if maxlen is not None and truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list/array was found')
            continue  # empty list/array was found
        if trunc_method == 'pre':
            trunc = s[-maxlen:]
        elif trunc_method == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % trunc_method)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if pad_method == 'post':
            x[idx, :len(trunc)] = trunc
        elif pad_method == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % pad_method)
    return x


def get_glove_matrix(glove_path, vocab, initial_embedding_np):
    """
    return a glove embedding matrix
    :param self:
    :param glove_file:
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    ef = open(glove_path, 'r', encoding='UTF-8')
    cnt = 0
    vec_array = initial_embedding_np
    old_avg = np.average(vec_array)
    old_std = np.std(vec_array)
    vec_array = vec_array.astype(np.float32)
    new_avg, new_std = 0, 0

    for line in ef.readlines():
        line = line.strip().split(' ')
        word, vec = line[0], line[1:]
        vec = np.array(vec, np.float32)
        if not vocab.has_word(word):
            continue
        word_idx = vocab.encode(word)
        if word_idx <vocab.vocab_size:
            cnt += 1
            vec_array[word_idx] = vec
            new_avg += np.average(vec)
            new_std += np.std(vec)
    new_avg /= cnt
    new_std /= cnt
    ef.close()
    logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (cnt, old_avg,
                                                                                          new_avg, old_std, new_std))
    return vec_array


def position_encoding_init(self, n_position, d_pos_vec):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
                             if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc


class ReplayBuffer(object):
    def __init__(self, batch_size=128, buffer_size=1e6): #turn_num = 1):
        #self.turn_num = turn_num # dataset loader reads based on total turn_num, so replay buffer use this too
        self.batch_size = batch_size
        self.max_size = int(buffer_size)

        self.ptr = 0
        self.crt_size = 0

        self.state = []
        self.action = []
        self.action_gt = []
        self.next_state = []
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, action_gt, next_state, reward, done):
        if self.crt_size < self.max_size:
            self.state.append(state)
            self.action.append(action)
            self.next_state.append(next_state)
            self.action_gt.append(action_gt)

        else:
            self.state[self.ptr] = state
            self.action[self.ptr] = action
            self.action_gt[self.ptr] = action_gt
            self.next_state[self.ptr] = next_state

        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def show(self, name, start=None, end=None):
        def show_helper(l, start=None, end=None):
            if start is not None and end is not None:
                for idx in range(start, end):
                    print(l[idx])
            else:
                for idx in range(self.crt_size):
                    print(l[idx])
        if name == 'state':
            print('********** state *********')
            show_helper(self.state, start, end)
        elif name == 'action': 
            print('********** action *********')
            show_helper(self.action, start, end)
        elif name == 'action_gt':
            print('********** action gt *********')
            show_helper(self.action_gt, start, end)
        elif name == 'next_state':
            print('********** next state *********')
            show_helper(self.next_state, start, end)
        elif name == 'reward':
            print('********** reward *********')
            print(self.reward[start:end])
        elif name == 'not_done':
            print('********** not done *********')
            print(self.not_done[start:end])
        print(f'len of buffer now: {self.crt_size}')

    '''
    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        batch_states = [self.state[i] for i in ind]
        batch_actions = [self.action[i] for i in ind]
        batch_actions_gt = [self.action_gt[i] for i in ind]
        batch_next_states = [self.next_state[i] for i in ind]
        batch_rewards = self.reward[ind]
        batch_not_dones = self.not_done[ind]

        # Convert batch_rewards and batch_not_dones to suitable formats if necessary
        return batch_states, batch_actions, batch_actions_gt, batch_next_states, batch_rewards, batch_not_dones
    '''
    def save(self, save_file):
        # Convert numpy arrays to lists for JSON compatibility
        rewards_list = self.reward[:self.crt_size].flatten().tolist()
        not_dones_list = self.not_done[:self.crt_size].flatten().tolist()
        
        # Create a dictionary of all the data
        save_dict = {
            'state': self.state[:self.crt_size],
            'action': self.action[:self.crt_size],
            'action_gt': self.action_gt[:self.crt_size], 
            'next_state': self.next_state[:self.crt_size],
            'reward': rewards_list,
            'not_done': not_dones_list,
            'ptr': self.ptr,
            'crt_size': self.crt_size
        }
        
        # Write the dictionary to a file as JSON
        with open(save_file, 'w') as f:
            json.dump(save_dict, f)
    
    def load(self, save_file):
        # Read the data from the file
        with open(save_file, 'r') as f:
            load_dict = json.load(f)
        
        # Update the buffer attributes
        self.state = load_dict['state']
        self.action = load_dict['action']
        self.action_gt = load_dict['action_gt']
        self.next_state = load_dict['next_state']
        self.reward = np.array(load_dict['reward']).reshape(-1, 1)
        self.not_done = np.array(load_dict['not_done']).reshape(-1, 1)
        self.ptr = load_dict['ptr']
        self.crt_size = load_dict['crt_size']

        print(f"Replay Buffer loaded with {self.crt_size} elements.")
                # 分割数据为训练集和验证集 (90% 训练, 10% 验证)
        indices = np.arange(len(self.state))
        train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)

        self.train_state = [self.state[i] for i in train_indices]
        self.train_action = [self.action[i] for i in train_indices]
        self.train_next_state = [self.next_state[i] for i in train_indices]
        self.train_reward = self.reward[train_indices]
        self.train_not_done = self.not_done[train_indices]
        self.train_size = len(self.train_state)

        self.val_state = [self.state[i] for i in val_indices]
        self.val_action = [self.action[i] for i in val_indices]
        self.val_next_state = [self.next_state[i] for i in val_indices]
        self.val_reward = self.reward[val_indices]
        self.val_not_done = self.not_done[val_indices]
        self.val_size = len(self.val_state)

        print(f"Data loaded. Train size: {self.train_size}, Validation size: {self.val_size}")

    def sample(self, batch_size, data_type='train'):
        if data_type == 'train':
            indices = np.random.randint(0, self.train_size, size=batch_size)
            batch_states = [self.train_state[i] for i in indices]
            batch_actions = [self.train_action[i] for i in indices]
            batch_next_states = [self.train_next_state[i] for i in indices]
            batch_rewards = self.train_reward[indices]
            batch_not_dones = self.train_not_done[indices]
        elif data_type == 'val':
            indices = np.random.randint(0, self.val_size, size=batch_size)
            batch_states = [self.val_state[i] for i in indices]
            batch_actions = [self.val_action[i] for i in indices]
            batch_next_states = [self.val_next_state[i] for i in indices]
            batch_rewards = self.val_reward[indices]
            batch_not_dones = self.val_not_done[indices]
        else:
            raise ValueError("data_type should be either 'train' or 'val'.")

        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_not_dones
def merge_turn_buffer(prev, crt, replay_buffer):
    '''
    Given the evaluation process prev and current bspn, aspn and gen respectively, merged it into 
    [prev state, prev act, cur state] for replay buffer

    Input:
    prev, crt: dict
        [dial_id][state/action/action_gen/reward]
    replay_buffer
        ReplayBuffer class
    '''
    done = float((prev == crt))
    for dial_id in prev:
        # state, action, action_gt, next_state, reward
        replay_buffer.add(prev[dial_id]['state'], prev[dial_id]['action_gen'], prev[dial_id]['action'], crt[dial_id]['state'], prev[dial_id]['reward'], done)
    return replay_buffer


def add_reward_buffer(reward_dict, buffer, turn_num):
    '''
    update reward info into buffer

    Input:
    turn_num:
        crt turn_num for idx in reward_dict
    reward_dict: 
        [dial_id][turn_num]['reward']
    buffer:
        [dial_id]['aspn/bspn/aspn_gen/bspn_gen']
    '''
    for dial_id in buffer:
        buffer[dial_id]['reward'] = reward_dict[dial_id][turn_num]['reward']
    return buffer