import os, random, argparse, time, logging, json, tqdm
import numpy as np

import torch
from torch.optim import Adam

import utils
from utils import ReplayBuffer, merge_turn_buffer, add_reward_buffer
from config import global_config as cfg
from reader import MultiWozReader
from damd_net import DAMD, BCQ,  cuda_, get_one_hot_input
from eval import MultiWozEvaluator
from torch.utils.tensorboard import SummaryWriter
from otherconfig import *
import shutil
import base64
import hashlib
import re
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import nvidia_smi
import pdb, datetime, random
from copy import deepcopy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
class Model(object):
    def __init__(self, bcq_model=None):
        self.reader = MultiWozReader()
        if len(cfg.cuda_device)==1:
            if bcq_model is not None:
                self.m = DAMD(self.reader, bcq_model)
            else:
                self.m =DAMD(self.reader)
        else:
            m = DAMD(self.reader)
            self.m=torch.nn.DataParallel(m, device_ids=cfg.cuda_device)
            # print(self.m.module)
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        if cfg.cuda: self.m = self.m.cuda()  #cfg.cuda_device[0]
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=5e-5)
        self.base_epoch = -1

        if cfg.limit_bspn_vocab:
            self.reader.bspn_masks_tensor = {}
            for key, values in self.reader.bspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.bspn_masks_tensor[key] = v_
        if cfg.limit_aspn_vocab:
            self.reader.aspn_masks_tensor = {}
            for key, values in self.reader.aspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.aspn_masks_tensor[key] = v_
                
        if cfg.enable_tensorboard:
            self.writer = SummaryWriter(cfg.tensorboard_path)
        self.epoch=0
        if other_config['gen_per_epoch_report']==True:
            self.df = pd.DataFrame(columns=['dial_id','success','match','bleu','rollout'])

    def add_torch_input(self, inputs, mode='train', first_turn=False):
        need_onehot = ['user', 'usdx', 'bspn', 'aspn', 'pv_resp', 'pv_bspn', 'pv_aspn',
                                   'dspn', 'pv_dspn', 'bsdx', 'pv_bsdx', 'cntfact_bspn', 'pv_cntfact_bspn', 'cntfact_bsdx', 'pv_cntfact_bsdx']
        inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())
        if cfg.enable_cntfact:
            input_keys = ['user', 'usdx', 'resp', 'bspn', 'aspn', 'bsdx', 'dspn', 'cntfact_bspn', 'cntfact_bsdx']
        else:
            input_keys = ['user', 'usdx', 'resp', 'bspn', 'aspn', 'bsdx', 'dspn']
        for item in input_keys:
            if not cfg.enable_aspn and item == 'aspn':
                continue
            if not cfg.enable_bspn and item == 'bspn':
                continue
            if not cfg.enable_dspn and item == 'dspn':
                continue
            if item in ['cntfact_bspn', 'cntfact_bsdx'] and cfg.enable_multi_cntfact:
                inputs[item] = []
                inputs[item+'_nounk'] = []
                #inputs[item+'_4loss'] = []
                inputs[item+'_onehot'] = []
                inputs['pv_'+item] = []
                inputs['pv_'+item+'_nounk'] = []
                inputs['pv_' + item + '_onehot'] = []
                for i in range(cfg.topk_cntfact):
                    inputs[item].append(cuda_(torch.from_numpy(inputs[item+'_unk_np'][i]).long()))
                    
                for i in range(cfg.topk_cntfact):
                    if item in ['user', 'usdx', 'resp', 'bspn', 'cntfact_bspn']:
                        inputs[item+'_nounk'].append(cuda_(torch.from_numpy(inputs[item+'_np'][i]).long()))
                    else:
                        inputs[item+'_nounk'].append(inputs[item][i])
                    # print(item, inputs[item].size())
                    if item in ['resp', 'bspn', 'aspn', 'bsdx', 'dspn', 'cntfact_bspn', 'cntfact_bsdx']:
                        if 'pv_'+item+'_unk_np' not in inputs:
                            continue
                        inputs['pv_'+item].append(cuda_(torch.from_numpy(inputs['pv_'+item+'_unk_np'][i]).long()))
                        if item in ['user', 'usdx', 'bspn', 'cntfact_bspn']:
                            inputs['pv_'+item+'_nounk'].append(cuda_(torch.from_numpy(inputs['pv_'+item+'_np'][i]).long()))
                            #inputs[item+'_4loss'].append(self.index_for_loss(item[i], inputs))
                        else:
                            inputs['pv_'+item+'_nounk'].append(inputs['pv_'+item][i])
                            #inputs[item+'_4loss'].append(inputs[item][i])
                        if 'pv_' + item in need_onehot:
                            inputs['pv_' + item + '_onehot'].append(get_one_hot_input(inputs['pv_'+item+'_unk_np'][i]))
                    if item in need_onehot:
                        inputs[item+'_onehot'].append(get_one_hot_input(inputs[item+'_unk_np'][i]))
            else:
                inputs[item] = cuda_(torch.from_numpy(inputs[item+'_unk_np']).long())
                if item in ['user', 'usdx', 'resp', 'bspn', 'cntfact_bspn']:
                    inputs[item+'_nounk'] = cuda_(torch.from_numpy(inputs[item+'_np']).long())
                else:
                    inputs[item+'_nounk'] = inputs[item]
                # print(item, inputs[item].size())
                if item in ['resp', 'bspn', 'aspn', 'bsdx', 'dspn', 'cntfact_bspn', 'cntfact_bsdx']:
                    if 'pv_'+item+'_unk_np' not in inputs:
                        continue
                    inputs['pv_'+item] = cuda_(torch.from_numpy(inputs['pv_'+item+'_unk_np']).long())
                    if item in ['user', 'usdx', 'bspn', 'cntfact_bspn']:
                        inputs['pv_'+item+'_nounk'] = cuda_(torch.from_numpy(inputs['pv_'+item+'_np']).long())
                        inputs[item+'_4loss'] = self.index_for_loss(item, inputs)
                    else:
                        inputs['pv_'+item+'_nounk'] = inputs['pv_'+item]
                        inputs[item+'_4loss'] = inputs[item]
                    if 'pv_' + item in need_onehot:
                        inputs['pv_' + item + '_onehot'] = get_one_hot_input(inputs['pv_'+item+'_unk_np'])
                if item in need_onehot:
                    inputs[item+'_onehot'] = get_one_hot_input(inputs[item+'_unk_np'])

        if cfg.multi_acts_training and 'aspn_aug_unk_np' in inputs:
            inputs['aspn_aug'] = cuda_(torch.from_numpy(inputs['aspn_aug_unk_np']).long())
            inputs['aspn_aug_4loss'] = inputs['aspn_aug']

        if 'G_unk_np' in inputs:
            inputs['G'] = cuda_(torch.from_numpy(inputs['G_unk_np']))
        if 'Q_unk_np' in inputs:
            inputs['Q'] = cuda_(torch.from_numpy(inputs['Q_unk_np']))
        if 'bhProb_unk_np' in inputs:
            inputs['bhProb'] = cuda_(torch.from_numpy(inputs['bhProb_unk_np']))
        return inputs

    def index_for_loss(self, item, inputs):
        raw_labels = inputs[item+'_np']
        if item == 'bspn':
            copy_sources = [inputs['user_np'], inputs['pv_resp_np'], inputs['pv_bspn_np']]
        elif item == 'cntfact_bspn':
            copy_sources = [inputs['user_np'], inputs['pv_resp_np'], inputs['pv_cntfact_bspn_np']]
        elif item == 'bsdx':
            copy_sources = [inputs['usdx_np'], inputs['pv_resp_np'], inputs['pv_bsdx_np']]
        elif item == 'cntfact_bsdx':
            copy_sources = [inputs['usdx_np'], inputs['pv_resp_np'], inputs['pv_cntfact_bsdx_np']]
        elif item == 'aspn':
            copy_sources = []
            if cfg.use_pvaspn:
                copy_sources.append(inputs['pv_aspn_np'])
            if cfg.enable_bspn:
                copy_sources.append(inputs[cfg.bspn_mode+'_np'])
        elif item == 'dspn':
            copy_sources = [inputs['pv_dspn_np']]
        elif item == 'resp':
            copy_sources = [inputs['usdx_np']]
            if cfg.enable_bspn:
                copy_sources.append(inputs[cfg.bspn_mode+'_np'])
            if cfg.enable_aspn:
                copy_sources.append(inputs['aspn_np'])
        else:
            return
        new_labels = np.copy(raw_labels)
        if copy_sources:
            bidx, tidx = np.where(raw_labels>=self.reader.vocab_size)
            copy_sources = np.concatenate(copy_sources, axis=1)
            for b in bidx:
                for t in tidx:
                    oov_idx = raw_labels[b, t]
                    if len(np.where(copy_sources[b, :] == oov_idx)[0])==0:
                        new_labels[b, t] = 2
        return cuda_(torch.from_numpy(new_labels).long())
    def get_turn_G(self, ids, reward_dict=None, gamma=0.99, turn_num=0):
        '''
        based on each turn's reward, calculate Gain and return the inputs' format result 

        reward_dict[dial_id] = success, match, cntfact, reward
        '''
        Gs = []
        Rs = []
        for dial_id in ids: # make sure g matches correct dial id
            if turn_num == 0:
                reward_dict[dial_id][turn_num]['G'] = reward_dict[dial_id][turn_num]['reward']
            else: 
                reward_dict[dial_id][turn_num]['G'] = reward_dict[dial_id][turn_num]['reward'] + gamma * reward_dict[dial_id][turn_num - 1]['G']
            Gs.append(reward_dict[dial_id][turn_num]['G'])
            Rs.append(reward_dict[dial_id][turn_num]['reward'])
        return Rs, Gs
    def update_cntfact_r(self, ids, reward_dict=None, cntfact_penalty=-1):
        '''
        add cntfact reward into success and match reward if current turn enabled cntfact bspn
        '''
        for dial_id in ids:
            for turn_num in reward_dict[dial_id]:
                reward_dict[dial_id][turn_num]['cntfact'] = cntfact_penalty
                reward_dict[dial_id][turn_num]['reward'] += cntfact_penalty
        return reward_dict

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        weight_decay_count = cfg.weight_decay_count
        train_time = 0
        sw = time.time()         

        rl_step = 0
        if cfg.generate_bcq:
            replay_buffer = ReplayBuffer(cfg.batch_size)
        else:
            replay_buffer = None
        for epoch in range(cfg.epoch_num):
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            log_sup_loss = 0
            log_contrast_loss = 0
            sup_loss = 0
            sup_cnt = 0
            optim = self.optim
            # data_iterator generatation size: (batch num, turn num, batch size)
            btm = time.time()
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                hidden_states = {}
                batch_collection = {}
                each_dial_reward = {}
                if cfg.enable_cntfact:
                    py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None, 'pv_cntfact_bspn': None, 'pv_cntfact_bsdx': None}
                else:
                    py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None}
                bgt = time.time()
                if cfg.enable_cntfact_reward:
                    cntfact_index = random.randint(0, cfg.topk_cntfact-1)  # fix the same cntfact batch for an episode
                for turn_num, turn_batch in enumerate(dial_batch): # turn_batch: list of batch's number of dict, ['dial_id', 'user', 'usdx', 'resp', 'bspn', 'bsdx', 'aspn', 'dspn', 'pointer', 'input_pointer', 'turn_domain', 'turn_num']
                    # print('turn %d'%turn_num)
                    # print(len(turn_batch['dial_id'])) 
                    optim.zero_grad()
                    first_turn = (turn_num==0)
                    if cfg.enable_cntfact_reward: #TODO: test a batch or a turn's cntfact reward better
                        cntfact_activate = (cfg.sample_ratio >= random.random())
                    else:
                        cntfact_activate = False
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn) 
                    if cfg.enable_cntfact_reward and cntfact_activate:
                        inputs[cfg.bspn_mode+'_np']= inputs[cfg.cntfact_bspn_mode+'_np'][cntfact_index]
                        inputs[cfg.bspn_mode+'_unk_np']= inputs[cfg.cntfact_bspn_mode+'_unk_np'][cntfact_index]
                    # current inputs:
                    # ['pv_resp_np', 'pv_resp_unk_np', 'pv_bspn_np', 'pv_bspn_unk_np', 'pv_aspn_np', 'pv_aspn_unk_np', 'pv_dspn_np', 'pv_dspn_unk_np', 'pv_bsdx_np', 
                    # 'pv_bsdx_unk_np', 'user_np', 'user_unk_np', 'usdx_np', 'usdx_unk_np', 'resp_np', 'resp_unk_np', 'bspn_np', 'bspn_unk_np', 'aspn_np', 'aspn_unk_np', 
                    # 'bsdx_np', 'bsdx_unk_np', 'db_np', 'turn_domain', 'dial_id', 'turn_num']
                    # for k,v in inputs.items():
                    #     if k in ["turn_domain",'db_np']:
                    #         print(f"{k}: {v[0]}")
                    #     else:
                    #         print(f"{k}: {self.reader.vocab.sentence_decode(v[0])}")
                    # if turn_num==1:
                    #     exit(0)
                    inputs = self.add_torch_input(inputs, first_turn=first_turn)

                    # current inputs:
                    # ['pv_resp_np', 'pv_resp_unk_np', 'pv_bspn_np', 'pv_bspn_unk_np', 'pv_aspn_np', 'pv_aspn_unk_np', 'pv_dspn_np', 'pv_dspn_unk_np', 'pv_bsdx_np', 
                    # 'pv_bsdx_unk_np', 'user_np', 'user_unk_np', 'usdx_np', 'usdx_unk_np', 'resp_np', 'resp_unk_np', 'bspn_np', 'bspn_unk_np', 'aspn_np', 'aspn_unk_np', 
                    # 'bsdx_np', 'bsdx_unk_np', 'db_np', 'turn_domain', 'dial_id', 'turn_num', 'db', 'user', 'user_nounk', 'user_onehot', 'usdx', 'usdx_nounk', 'usdx_onehot', 
                    # 'resp', 'resp_nounk', 'pv_resp', 'pv_resp_nounk', 'resp_4loss', 'pv_resp_onehot', 'bspn', 'bspn_nounk', 'pv_bspn', 'pv_bspn_nounk', 'bspn_4loss', 
                    # 'pv_bspn_onehot', 'bspn_onehot', 'aspn', 'aspn_nounk', 'pv_aspn', 'pv_aspn_nounk', 'aspn_4loss', 'pv_aspn_onehot', 'aspn_onehot', 'bsdx', 'bsdx_nounk', 
                    # 'pv_bsdx', 'pv_bsdx_nounk', 'bsdx_4loss', 'pv_bsdx_onehot', 'bsdx_onehot'])
                    # total_loss, losses, hidden_states = self.m(inputs, hidden_states, first_turn, mode='train')
                    if cfg.enable_rl:
                        total_loss, losses, probs, decoded = self.m(inputs, hidden_states, first_turn, mode='rl_supervised')
                    else:
                        total_loss, losses = self.m(inputs, hidden_states, first_turn, mode='train')
                    # print('forward completed')
                    py_prev['pv_resp'] = turn_batch['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn']
                    if cfg.enable_cntfact and not cfg.enable_contrast:
                        py_prev['pv_cntfact_bsdx'] = turn_batch['cntfact_bsdx']
                        py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn']
                    if cfg.enable_contrast:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                        py_prev['pv_cntfact_bsdx'] = turn_batch['cntfact_bsdx']
                        py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn']

                    if cfg.enable_rl:
                        turn_batch['resp_gen'] = decoded['resp']
                        if cfg.bspn_mode == 'bspn' or cfg.enable_dst: #False
                            if cfg.enable_cntfact and cfg.cntfact_bspn_mode == 'cntfact_bspn' and not cfg.enable_contrast and not cfg.enable_cntfact_reward:
                                turn_batch['bspn_gen'] = decoded['cntfact_bspn']#decoded['bspn']
                            else:
                                turn_batch['bspn_gen'] = decoded['bspn']
                        if cfg.enable_aspn:
                            turn_batch['aspn_gen'] = decoded['aspn']

                        turn_batch['pi(a|b)'] = probs['pi(a|b)']
                        batch_collection.update(self.reader.inverse_transpose_batch(dial_batch))
                        turn_results, _ = self.reader.wrap_result(batch_collection)
                        '''
                        rollouts = {}
                        for row in results:
                            # turn each turn list into each dial list
                            if row['dial_id'] not in rollouts:
                                rollouts[row['dial_id']]={}
                            rollout = rollouts[row['dial_id']]
                            rollout_step = {}
                            rollout[row['turn_num']] = rollout_step
                            
                            rollout_step['resp'] = row['resp']
                            rollout_step['resp_gen'] = row['resp_gen']
                            rollout_step['aspn'] = row['aspn']
                            
                            #TODO: add cntfact here, which restored in reward_report.csv
                            if 'bspn' in row:
                                rollout_step['bspn'] = row['bspn']
                            if 'bspn_gen' in row:
                                rollout_step['bspn_gen'] = row['bspn_gen']
                            
                            rollout_step['aspn_gen'] = row['aspn_gen']

                            # eval need dspn and dspn_gen
                            rollout_step['dspn'] = turn_batch['dspn']
                            #rollout_step['dspn_gen'] = row['dspn_gen']

                        if other_config['gen_per_epoch_report']==True:
                            bleu, success, match,req_offer_counts,stats,all_true_reqs,all_pred_reqs, success_true, all_successes, all_matches, all_bleus ,dial_ids = self.evaluator.validation_metric(results, return_rich=True, return_per_dialog=True,soft_acc=other_config['soft_acc'])
                            for i,dial_id in enumerate(dial_ids):
                                self.df.loc[len(self.df)] = [dial_id,all_successes[i],all_matches[i],all_bleus[i],json.dumps(rollouts[dial_id])]
                            self.df.to_csv(other_config['per_epoch_report_path'])
                        else:
                        '''

                        use_contrast = cntfact_activate and cfg.enable_contrast_reward
                        if cfg.generate_bcq:
                            bleu, success, match, each_dial_reward, crt_turn_buffer = self.evaluator.validation_metric(turn_results, return_rich=False, return_each=True, return_dict=each_dial_reward, turn_num=turn_num, return_contrast=use_contrast, return_buffer=True)

                        else:
                            bleu, success, match, each_dial_reward = self.evaluator.validation_metric(turn_results, return_rich=False, return_each=True, return_dict=each_dial_reward, turn_num=turn_num, return_contrast=use_contrast)
                        #bleu, success, match, each_dial_reward = self.evaluator.validation_metric(turn_results, return_rich=False, return_each=True, return_dict=each_dial_reward, turn_num=turn_num, return_contrast=cntfact_active)
                        if cntfact_activate:
                            each_dial_reward = self.update_cntfact_r(turn_batch['dial_id'], each_dial_reward, cfg.cntfact_penalty)
                        reward_log, each_dial_gain = self.get_turn_G(turn_batch['dial_id'], each_dial_reward, cfg.gamma, turn_num)
                        # TODO: use sentinel to save prev_turn_buffer, write a func to merge the two, and update in replay_buffer
                        if cfg.generate_bcq:
                            crt_turn_buffer = add_reward_buffer(each_dial_reward, crt_turn_buffer, turn_num)
                            # initiate a replay buffer

                            #TODO: move init and update after buffer merge
                            if turn_num > 0:
                                replay_buffer = merge_turn_buffer(prev_turn_buffer, crt_turn_buffer, replay_buffer)
                            prev_turn_buffer = crt_turn_buffer
                            if turn_num == len(dial_batch) - 1:
                                replay_buffer = merge_turn_buffer(prev_turn_buffer, crt_turn_buffer, replay_buffer)
                        #print(f'current iter: {iter_num} turn: {turn_num} reward metrics: success {success}, match {match}')
                        #print('current each dial reward:')
                        #print(each_dial_reward)
                        # TODO: G calculation 
                        #print(f'each turn Gain: {each_dial_gain}')
                        inputs['G_np'] = np.array(each_dial_gain)
                        inputs['G_unk_np'] = inputs['G_np']
                        inputs['G'] = cuda_(torch.from_numpy(inputs['G_unk_np']))
                        # TODO: policy gradiant 
                        policy_loss, losses = self.m(inputs, hidden_states, first_turn, mode='rl_policy', losses=losses, probs=probs)
                        total_loss += policy_loss
                        #if cfg.enable_rl and cfg.enable_tensorboard:
                            #print(f'********** iter {iter_num} turn {turn_num} ***********')
                            #print(f'epoch reward {reward_log}')
                            #step_log_reward = sum(reward_log) / len(reward_log)
                            #print(f'average reward {step_log_reward}')
                            #self.writer.add_scalar('G/rl', step_log_reward, rl_step)
                            #rl_step += 1

                    total_loss = total_loss.mean()
                    # print('forward time:%f'%(time.time()-test_begin))
                    # test_begin = time.time()
                    total_loss.backward(retain_graph=False)
                    # total_loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    # print('backward time:%f'%(time.time()-test_begin))
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    optim.step()
                    sup_loss += float(total_loss)
                    if cfg.enable_tensorboard:
                        log_contrast_loss += 0 if not cfg.enable_contrast else float(losses['contrast'])
                        log_sup_loss = sup_loss - log_contrast_loss
                    sup_cnt += 1
                    torch.cuda.empty_cache()


                #nupdated after all turn finished
                '''
                results, _ = self.reader.wrap_result(batch_collection)
                rollouts = {}
                for row in results:
                    # turn each turn list into each dial list
                    if row['dial_id'] not in rollouts:
                        rollouts[row['dial_id']]={}
                    rollout = rollouts[row['dial_id']]
                    rollout_step = {}
                    rollout[row['turn_num']] = rollout_step
                    
                    rollout_step['resp'] = row['resp']
                    rollout_step['resp_gen'] = row['resp_gen']
                    rollout_step['aspn'] = row['aspn']
                    
                    #TODO: add cntfact here, which restored in reward_report.csv
                    if 'bspn' in row:
                        rollout_step['bspn'] = row['bspn']
                    if 'bspn_gen' in row:
                        rollout_step['bspn_gen'] = row['bspn_gen']
                    
                    rollout_step['aspn_gen'] = row['aspn_gen']

                    # eval need dspn and dspn_gen
                    rollout_step['dspn'] = turn_batch['dspn']
                    #rollout_step['dspn_gen'] = row['dspn_gen']

                if other_config['gen_per_epoch_report']==True:
                    bleu, success, match,req_offer_counts,stats,all_true_reqs,all_pred_reqs, success_true, all_successes, all_matches, all_bleus ,dial_ids = self.evaluator.validation_metric(results, return_rich=True, return_per_dialog=True,soft_acc=other_config['soft_acc'])
                    for i,dial_id in enumerate(dial_ids):
                        self.df.loc[len(self.df)] = [dial_id,all_successes[i],all_matches[i],all_bleus[i],json.dumps(rollouts[dial_id])]
                    self.df.to_csv(other_config['per_epoch_report_path'])
                else:
                    bleu, success, match,req_offer_counts,stats,all_true_reqs,all_pred_reqs, success_true = self.evaluator.validation_metric(results, return_rich=True)
                '''
                if (iter_num+1)%cfg.report_interval==0:
                    if cfg.enable_cntfact:
                        if cfg.enable_contrast:
                            logging.info(
                                    'iter:{} [total|cntfact_bspn|bspn|aspn|resp|contrast] loss: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} grad:{:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                            float(total_loss),
                                                                            float(losses[cfg.cntfact_bspn_mode]), float(losses[cfg.bspn_mode]), float(losses['aspn']),float(losses['resp']),float(losses['contrast']),
                                                                            grad,
                                                                            time.time()-btm,
                                                                            turn_num+1))
                        elif cfg.enable_cntfact_reward:
                            logging.info(
                                    'iter:{} [total|reward|bspn|aspn|resp] loss: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} grad:{:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                            float(total_loss),
                                                                            float(sum(each_dial_gain) / len(each_dial_gain)), float(losses[cfg.bspn_mode]), float(losses['aspn']),float(losses['resp']),
                                                                            grad,
                                                                            time.time()-btm,
                                                                            turn_num+1))

                        else:
                            logging.info(
                                    'iter:{} [total|cntfact_bspn|bspn|aspn|resp] loss: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} grad:{:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                            float(total_loss),
                                                                            float(losses[cfg.cntfact_bspn_mode]), float(losses[cfg.bspn_mode]), float(losses['aspn']),float(losses['resp']),
                                                                            grad,
                                                                            time.time()-btm,
                                                                            turn_num+1))


                    else:
                        logging.info(
                                'iter:{} [total|bspn|aspn|resp] loss: {:.2f} {:.2f} {:.2f} {:.2f} grad:{:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                            float(total_loss),
                                                                            float(losses[cfg.bspn_mode]),float(losses['aspn']),float(losses['resp']),
                                                                            grad,
                                                                            time.time()-btm,
                                                                            turn_num+1))
                    if cfg.enable_dst and cfg.bspn_mode == 'bsdx':
                        logging.info('bspn-dst:{:.3f}'.format(float(losses['bspn'])))
                    if cfg.multi_acts_training:
                        logging.info('aspn-aug:{:.3f}'.format(float(losses['aspn_aug'])))
                    if cfg.generate_bcq:
                        logging.info( 'len of buffer now{}'.format(replay_buffer.crt_size))

                # btm = time.time()
                # if (iter_num+1)%40==0:
                #     print('validation checking ... ')
                #     valid_sup_loss, valid_unsup_loss = self.validate(do_test=False)
                #     logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            
            if cfg.enable_tensorboard:
                epoch_log_sup_loss = log_sup_loss / (sup_cnt + 1e-8)
                epoch_log_contrast_loss = log_contrast_loss / (sup_cnt + 1e-8)
                epoch_log_total_loss = epoch_log_sup_loss + epoch_log_contrast_loss
                self.writer.add_scalar('SupervisedLoss/train', epoch_log_sup_loss, epoch)
                self.writer.add_scalar('ContrastLoss/train', epoch_log_contrast_loss, epoch)
                self.writer.add_scalar('TotalLoss/train', epoch_log_total_loss, epoch)
                if cfg.enable_rl:
                    epoch_log_gain = sum(each_dial_gain) / len(each_dial_gain)
                    self.writer.add_scalar('G/rl', epoch_log_gain, epoch)

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8) # ori loss
            # do_test = True if (epoch+1)%5==0 else False
            do_test = False
            valid_loss = self.validate(do_test=do_test)
            if cfg.enable_tensorboard:
                self.writer.add_scalar('Score/valid', 130 - valid_loss, epoch)
            logging.info('epoch: %d, sup loss: %.3f, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, sup_loss, epoch_sup_loss,
                    valid_loss, (time.time()-sw)/60))
            # self.save_model(epoch)
            if valid_loss <= prev_min_loss:
                early_stop_count = cfg.early_stop_count
                weight_decay_count = cfg.weight_decay_count
                prev_min_loss = valid_loss
                self.save_model(epoch)
            else:
                early_stop_count -= 1
                weight_decay_count -= 1
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))
                if not early_stop_count:
                    self.load_model()
                    if cfg.generate_bcq:
                        print('******* saved buffer *********')
                        buffer_path = os.path.join(cfg.exp_path, 'replay.json')
                        replay_buffer.save(buffer_path)
                    print('result preview...')
                    file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'eval_log%s.json'%cfg.seed))
                    logging.getLogger('').addHandler(file_handler)
                    logging.info(str(cfg))
                    self.eval()
                    return
                if not weight_decay_count:
                    lr *= cfg.lr_decay
                    self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                                  weight_decay=5e-5)
                    weight_decay_count = cfg.weight_decay_count
                    logging.info('learning rate decay, learning rate: %f' % (lr))
        self.load_model()
        if cfg.generate_bcq:
            print('******* saved buffer *********')
            buffer_path = os.path.join(cfg.exp_path, 'replay.json')
            replay_buffer.save(buffer_path)
        print('result preview...')
        file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'eval_log%s.json'%cfg.seed))
        logging.getLogger('').addHandler(file_handler)
        logging.info(str(cfg))
        self.eval()


    def validate(self, data='dev', do_test=False):
        print("************* validate test **********")
        #pdb.set_trace()
        self.m.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_batches(data)
        result_collection = {}
        for batch_num, dial_batch in enumerate(data_iterator):
            hidden_states = {}
            if cfg.enable_cntfact:
                py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None, 'pv_cntfact_bspn': None, 'pv_cntfact_bsdx': None}
            else:
                py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                inputs = self.add_torch_input(inputs, first_turn=first_turn)
                # total_loss, losses, hidden_states = self.m(inputs, hidden_states, first_turn, mode='train')
                if cfg.valid_loss not in ['score', 'match', 'success', 'bleu']:
                    total_loss, losses = self.m(inputs, hidden_states, first_turn, mode='train')
                    py_prev['pv_resp'] = turn_batch['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn']
                    if cfg.enable_cntfact and not cfg.enable_contrast:
                        py_prev['pv_cntfact_bsdx'] = turn_batch['cntfact_bsdx']
                        py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn']
                    if cfg.enable_contrast:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                        py_prev['pv_cntfact_bsdx'] = turn_batch['cntfact_bsdx']
                        py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn']

                    if cfg.valid_loss == 'total_loss':
                        valid_loss += float(total_loss)
                    elif cfg.valid_loss == 'bspn_loss':
                        valid_loss += float(losses[cfg.bspn_mode])
                    elif cfg.valid_loss == 'aspn_loss':
                        valid_loss += float(losses['aspn'])
                    elif cfg.valid_loss == 'resp_loss':
                        valid_loss += float(losses['reps'])
                    else:
                        raise ValueError('Invalid validation loss type!')
                else:
                    decoded = self.m(inputs, hidden_states, first_turn, mode='test')
                    turn_batch['resp_gen'] = decoded['resp']
                    if cfg.bspn_mode == 'bspn' or cfg.enable_dst: #False
                        if cfg.enable_cntfact and cfg.cntfact_bspn_mode == 'cntfact_bspn' and not cfg.enable_contrast and not cfg.enable_cntfact_reward:
                            turn_batch['bspn_gen'] = decoded['cntfact_bspn']#decoded['bspn']
                        else:
                            turn_batch['bspn_gen'] = decoded['bspn']
                    if cfg.enable_aspn:
                        turn_batch['aspn_gen'] = decoded['aspn']
                    py_prev['pv_resp'] = turn_batch['resp'] if cfg.use_true_pv_resp else decoded['resp']
                    if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
                        py_prev['pv_'+cfg.bspn_mode] = turn_batch[cfg.bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.bspn_mode] # py_prev['pv_bsdx'] = turn_batch['bsdx']
                        py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn'] # True
                    elif cfg.enable_cntfact and not cfg.enable_contrast:
                        py_prev['pv_cntfact_bsdx'] = turn_batch['cntfact_bsdx'] if cfg.use_true_prev_bspn or 'cntfact_bsdx' not in decoded else decoded['cntfact_bsdx'] 
                        py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn'] if cfg.use_true_prev_bspn or 'cntfact_bspn' not in decoded else decoded['cntfact_bspn']
                    elif cfg.enable_contrast:
                        py_prev['pv_bsdx'] = turn_batch['bsdx'] if cfg.use_true_prev_bspn or 'bsdx' not in decoded else decoded['bsdx'] 
                        py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn']
                        py_prev['pv_cntfact_bsdx'] = turn_batch['cntfact_bsdx'] if cfg.use_true_prev_bspn or 'cntfact_bsdx' not in decoded else decoded['cntfact_bsdx'] 
                        py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn'] if cfg.use_true_prev_bspn or 'cntfact_bspn' not in decoded else decoded['cntfact_bspn']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn'] if cfg.use_true_prev_dspn else decoded['dspn']
                count += 1
                torch.cuda.empty_cache()

            if cfg.valid_loss in ['score', 'match', 'success', 'bleu']:
                result_collection.update(self.reader.inverse_transpose_batch(dial_batch))


        if cfg.valid_loss not in ['score', 'match', 'success', 'bleu']:
            valid_loss /= (count + 1e-8)
        else:
            results, _ = self.reader.wrap_result(result_collection) # ['dial_id', 'turn_num', 'user', 'bsdx_gen', 'bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn', 'dspn_gen', 'dspn', 'bspn', 'pointer']
            rollouts = {}
            for row in results:
                # turn each turn list into each dial list
                if row['dial_id'] not in rollouts:
                    rollouts[row['dial_id']]={}
                rollout = rollouts[row['dial_id']]
                rollout_step = {}
                rollout[row['turn_num']] = rollout_step
                
                rollout_step['resp'] = row['resp']
                rollout_step['resp_gen'] = row['resp_gen']
                rollout_step['aspn'] = row['aspn']
                
                #TODO: add cntfact here, which restored in reward_report.csv
                if 'bspn' in row:
                    rollout_step['bspn'] = row['bspn']
                if 'bspn_gen' in row:
                    rollout_step['bspn_gen'] = row['bspn_gen']
                
                rollout_step['aspn_gen'] = row['aspn_gen']

                    
            if other_config['gen_per_epoch_report']==True:
                bleu, success, match,req_offer_counts,stats,all_true_reqs,all_pred_reqs, success_true, all_successes, all_matches, all_bleus ,dial_ids = self.evaluator.validation_metric(results, return_rich=True, return_per_dialog=True,soft_acc=other_config['soft_acc'])
                for i,dial_id in enumerate(dial_ids):
                    self.df.loc[len(self.df)] = [dial_id,all_successes[i],all_matches[i],all_bleus[i],json.dumps(rollouts[dial_id])]
                self.df.to_csv(other_config['per_epoch_report_path'])
            else:
                bleu, success, match,req_offer_counts,stats,all_true_reqs,all_pred_reqs, success_true = self.evaluator.validation_metric(results, return_rich=True)
            
            score = 0.5 * (success + match) + bleu
            valid_loss = 130 - score
            logging.info('validation [CTR] match: %2.1f  success: %2.1f  bleu: %2.1f'%(match, success, bleu))
        self.m.train()
        if do_test:
            print('result preview...')
            self.eval()
        return valid_loss

    def eval(self, data='test'):
        print("*********** eval test ***********")
        #pdb.set_trace()
        self.m.eval()
        self.reader.result_file = None
        result_collection = {}
        data_iterator = self.reader.get_batches(data)
        for batch_num, dial_batch in tqdm.tqdm(enumerate(data_iterator)):
            # quit()
            # if batch_num > 0:
            #     continue
            hidden_states = {}
            if cfg.enable_cntfact:
                py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None, 'pv_cntfact_bspn': None, 'pv_cntfact_bsdx': None}
            else:
                py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None}
            print('batch_size:', len(dial_batch[0]['resp']))
            for turn_num, turn_batch in enumerate(dial_batch):
                # print('turn %d'%turn_num)
                # if turn_num!=0 and turn_num<4:
                #     continue
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                inputs = self.add_torch_input(inputs, first_turn=first_turn)
                decoded = self.m(inputs, hidden_states, first_turn, mode='test')
                #print(decoded)
                turn_batch['resp_gen'] = decoded['resp']
                #if cfg.bspn_mode == 'bsdx':
                if cfg.bspn_mode == 'bsdx' and not cfg.enable_cntfact:
                    turn_batch['bsdx_gen'] = decoded['bsdx'] if cfg.enable_bspn else [[0]] * len(decoded['resp']) #TODO: verify the effect of bsdx, bspn here.
                elif cfg.enable_cntfact and cfg.cntfact_bspn_mode == 'cntfact_bsdx' and not cfg.enable_contrast and not cfg.enable_cntfact_reward:
                    turn_batch['bsdx_gen'] = decoded['cntfact_bsdx'] if cfg.enable_bspn else [[0]] * len(decoded['resp']) #TODO: verify the effect of bsdx, bspn here.
                elif cfg.bspn_mode == 'bspn' or cfg.enable_dst:
                    if cfg.enable_cntfact and cfg.cntfact_bspn_mode == 'cntfact_bspn' and not cfg.enable_contrast and not cfg.enable_cntfact_reward:
                        turn_batch['bspn_gen'] = decoded['cntfact_bspn'] if cfg.enable_bspn else [[0]] * len(decoded['resp'])
                    else:
                        turn_batch['bspn_gen'] = decoded['bspn'] if cfg.enable_bspn else [[0]] * len(decoded['resp'])
                turn_batch['aspn_gen'] = decoded['aspn'] if cfg.enable_aspn else [[0]] * len(decoded['resp'])
                turn_batch['dspn_gen'] = decoded['dspn'] if cfg.enable_dspn else [[0]] * len(decoded['resp'])

                if self.reader.multi_acts_record is not None:
                    turn_batch['multi_act_gen'] = self.reader.multi_acts_record
                if cfg.record_mode:
                    turn_batch['multi_act'] = self.reader.aspn_collect
                    turn_batch['multi_resp'] = self.reader.resp_collect
                # print(turn_batch['user'])
                # print('user:', self.reader.vocab.sentence_decode(turn_batch['user'][0] , eos='<eos_u>', indicate_oov=True))
                # print('resp:', self.reader.vocab.sentence_decode(decoded['resp'][0] , eos='<eos_r>', indicate_oov=True))
                # print('bspn:', self.reader.vocab.sentence_decode(decoded['bspn'][0] , eos='<eos_b>', indicate_oov=True))
                # for b in range(len(decoded['resp'])):
                #     for i in range(5):
                #         print('aspn:', self.reader.vocab.sentence_decode(decoded['aspn'][i][b] , eos='<eos_a>', indicate_oov=True))

                py_prev['pv_resp'] = turn_batch['resp'] if cfg.use_true_pv_resp else decoded['resp']
                if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
                    py_prev['pv_'+cfg.bspn_mode] = turn_batch[cfg.bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.bspn_mode]
                    py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn']
                elif cfg.enable_cntfact and not cfg.enable_contrast:
                    py_prev['pv_'+cfg.cntfact_bspn_mode] = turn_batch[cfg.cntfact_bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.cntfact_bspn_mode]
                    py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn'] if cfg.use_true_prev_bspn or 'cntfact_bspn' not in decoded else decoded['cntfact_bspn']
                elif cfg.enable_contrast:
                    py_prev['pv_'+cfg.bspn_mode] = turn_batch[cfg.bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.bspn_mode]
                    py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn']
                    #py_prev['pv_'+cfg.cntfact_bspn_mode] = turn_batch[cfg.cntfact_bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.cntfact_bspn_mode]
                    #py_prev['pv_cntfact_bspn'] = turn_batch['cntfact_bspn'] if cfg.use_true_prev_bspn or 'cntfact_bspn' not in decoded else decoded['cntfact_bspn']
                if cfg.enable_aspn:
                    py_prev['pv_aspn'] = turn_batch['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                if cfg.enable_dspn:
                    py_prev['pv_dspn'] = turn_batch['dspn'] if cfg.use_true_prev_dspn else decoded['dspn']
                torch.cuda.empty_cache()
                # prev_z = turn_batch['bspan']
            # print('test iter %d'%(batch_num+1))
            # print(dial_batch)
            result_collection.update(self.reader.inverse_transpose_batch(dial_batch)) # ['dial_id', 'user', 'usdx', 'resp', 'bspn', 'bsdx', 'aspn', 'dspn', 'cntfact_bspn', 'cntfact_bsdx', 'pointer', 'input_pointer', 'turn_domain', 'turn_num', 'resp_gen', 'bspn_gen', 'aspn_gen', 'dspn_gen']

        # self.reader.result_file.close()
        if cfg.record_mode:
            self.reader.record_utterance(result_collection)
            quit()
        
        results, field = self.reader.wrap_result(result_collection)
        #print(results)
        self.reader.save_result('w', results, field)

        metric_results = self.evaluator.run_metrics(results)
        metric_field = list(metric_results[0].keys())
        req_slots_acc = metric_results[0]['req_slots_acc']
        info_slots_acc = metric_results[0]['info_slots_acc']

        self.reader.save_result('w', metric_results, metric_field,
                                            write_title='EVALUATION RESULTS:')
        self.reader.save_result('a', [info_slots_acc], list(info_slots_acc.keys()),
                                            write_title='INFORM ACCURACY OF EACH SLOTS:')
        self.reader.save_result('a', [req_slots_acc], list(req_slots_acc.keys()),
                                            write_title='REQUEST SUCCESS RESULTS:')
        self.reader.save_result('a', results, field+['wrong_domain', 'wrong_act', 'wrong_inform'],
                                            write_title='DECODED RESULTS:')
        self.reader.save_result_report(metric_results)
        # self.reader.metric_record(metric_results)
        self.m.train()
        return None

    def save_model(self, epoch, path=None, critical=False):
        if not cfg.save_log:
            return
        if not path:
            path = cfg.model_path
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)
        logging.info('Model saved')

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location='cpu')
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)
        logging.info('Model loaded')

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        if not cfg.multi_gpu:
            initial_arr = self.m.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.embedding.weight.data.copy_(emb)
        else:
            initial_arr = self.m.module.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.module.embedding.weight.data.copy_(emb)


    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        print('total trainable params: %d' % param_cnt)
        return param_cnt


class BCQModel(object):
    def __init__(self):
        self.reader = MultiWozReader()
        if torch.cuda.is_available() and cfg.cuda:
            device_id = cfg.cuda_device[0]  # 假设cfg.cuda_device是一个列表，我们使用列表中的第一个设备
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        self.device = device
        if len(cfg.cuda_device)==1:
            self.m = BCQ(self.reader, device=self.device).to(self.device)
        else:
            m = BCQ(self.reader).to(self.device)
            self.m=torch.nn.DataParallel(m, device_ids=cfg.cuda_device)
            # print(self.m.module)
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        #if cfg.cuda: self.m = self.m.cuda()  #cfg.cuda_device[0]
        #self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=5e-5)
        self.base_epoch = -1
                
        if cfg.enable_tensorboard:
            self.writer = SummaryWriter(cfg.tensorboard_path)
        self.epoch=0

    def add_torch_input(self, inputs, mode='train', first_turn=False):
        need_onehot = ['state', 'action', 'next_state']

        input_keys = ['state', 'action', 'next_state', 'reward', 'not_done']
        for item in input_keys:
            inputs[item] = cuda_(torch.from_numpy(inputs[item+'_unk_np']).long())
            if item in ['state', 'next_state', 'action']:
                inputs[item+'_nounk'] = cuda_(torch.from_numpy(inputs[item+'_np']).long())
            else:
                inputs[item+'_nounk'] = inputs[item]
            
            #if item in need_onehot:
            #    inputs[item+'_onehot'] = get_one_hot_input(inputs[item+'_unk_np'])

        return inputs


    def tokenize_sentence(self, batch_sentences, mode):
        tokenized = []
        for idx, sentence in enumerate(batch_sentences):
            tokenized.append(self.reader._get_buffer_data(sentence, mode))
        return tokenized
    
    '''
    def train_BCQ(self, state_dim, action_dim, max_action, device, args):
        # For saving files
        setting = f"{args.env}_{args.seed}"
        buffer_name = f"{args.buffer_name}_{setting}"

        # Initialize policy
        policy = BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

        # Load buffer
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
        replay_buffer.load(f"./buffers/{buffer_name}")
        
        evaluations = []
        episode_num = 0
        done = True 
        training_iters = 0
        
        while training_iters < args.max_timesteps: 
            pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/BCQ_{setting}", evaluations)

            training_iters += args.eval_freq
            print(f"Training iterations: {training_iters}")
    '''
    def generate_action(self, state):
        '''
        param:
        state: format as inputs['state']
        '''
        return self.m.select_action(state)

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        weight_decay_count = cfg.weight_decay_count
        train_time = 0
        sw = time.time()         

        rl_step = 0
        replay_buffer = ReplayBuffer(cfg.batch_size)
        replay_buffer.load(os.path.join(cfg.exp_path, 'replay.json'))
        for epoch in range(cfg.epoch_num):
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            loss_log = {'total':0, 'vae': 0, 'actor':0, 'critic': 0}
            cnt = 0
            # data_iterator generatation size: (batch num, turn num, batch size)
            btm = time.time()
            #data_iterator = self.reader.get_batches('train')
            for iter_num in range(replay_buffer.train_size // cfg.batch_size):
                bgt = time.time()
                batch_states, batch_actions, batch_next_states, batch_rewards, batch_not_dones = replay_buffer.sample(cfg.batch_size, data_type='train')
                inputs = {}
                batch_states = self.tokenize_sentence(batch_states, 'bspn')
                batch_actions = self.tokenize_sentence(batch_actions, 'aspn')
                batch_next_states = self.tokenize_sentence(batch_next_states, 'bspn')
                #TODO: see if set unknown matters here
                inputs['state_np'] = utils.padSeqs(batch_states, truncated=cfg.truncated, trunc_method='pre') 
                inputs['state_unk_np'] = deepcopy(inputs['state_np'])
                inputs['state_unk_np'][inputs['state_unk_np']>=self.reader.vocab_size] = 2   # <unk>
                inputs['action_np'] = utils.padSeqs(batch_actions, truncated=cfg.truncated, trunc_method='pre') 
                inputs['action_unk_np'] = deepcopy(inputs['action_np'])
                inputs['action_unk_np'][inputs['action_unk_np']>=self.reader.vocab_size] = 2   # <unk>
                inputs['next_state_np'] = utils.padSeqs(batch_next_states, truncated=cfg.truncated, trunc_method='pre') 
                inputs['next_state_unk_np'] = deepcopy(inputs['next_state_np'])
                inputs['next_state_unk_np'][inputs['next_state_unk_np']>=self.reader.vocab_size] = 2   # <unk>
                inputs['reward_unk_np'] = batch_rewards
                inputs['not_done_unk_np'] = batch_not_dones

                inputs = self.add_torch_input(inputs)

                vae_loss, actor_loss, critic_loss = self.m.train_forward(inputs['state'], inputs['action'], inputs['next_state'], inputs['reward'], inputs['not_done'], batch_size=cfg.batch_size)
                total_loss = vae_loss + actor_loss + critic_loss

                torch.cuda.empty_cache()

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info('iter:{} [vae|actor|critic] loss: {:.2f} {:.2f} {:.2f}  time: {:.1f}  '.format(iter_num+1,
                                                                            float(vae_loss),
                                                                            float(actor_loss), 
                                                                            float(critic_loss),
                                                                            time.time()-btm,
                                                                            ))
                loss_log['total'] += float(total_loss)
                loss_log['vae'] += float(vae_loss)
                loss_log['actor'] += float(actor_loss)
                loss_log['critic'] += float(critic_loss)
                cnt += 1
                torch.cuda.empty_cache()

            
            if cfg.enable_tensorboard:
                epoch_log_total_loss = loss_log['total'] / (cnt + 1e-8)
                epoch_log_vae_loss = loss_log['vae'] / (cnt + 1e-8)
                epoch_log_actor_loss = loss_log['actor'] / (cnt + 1e-8)
                epoch_log_critic_loss = loss_log['critic'] / (cnt + 1e-8)
                self.writer.add_scalar('Total BCQ Loss/train', epoch_log_total_loss, epoch)
                self.writer.add_scalar('VAE Loss/train', epoch_log_vae_loss, epoch)
                self.writer.add_scalar('Actor Loss/train', epoch_log_actor_loss, epoch)
                self.writer.add_scalar('Critic Loss/train', epoch_log_critic_loss, epoch)



            total_val_loss, vae_val_loss, actor_val_loss, critic_val_loss = self.validate(replay_buffer)
            if cfg.enable_tensorboard:
                self.writer.add_scalar('Total BCQ Loss/valid', total_val_loss, epoch)
                self.writer.add_scalar('VAE Loss/valid', vae_val_loss, epoch)
                self.writer.add_scalar('Actor Loss/valid', -actor_val_loss, epoch)
                self.writer.add_scalar('Critic Loss/valid', critic_val_loss, epoch)
            logging.info('epoch: %d, sup loss: %.3f, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, loss_log['total'], loss_log['total'] / (cnt + 1e-8),
                    total_val_loss, (time.time()-sw)/60))
            selected_act = self.m.select_action(inputs['state'])
            # self.save_model(epoch)
            if total_val_loss <= prev_min_loss:
                early_stop_count = cfg.early_stop_count
                prev_min_loss = total_val_loss
                self.save_model(epoch)
            else:
                early_stop_count -= 1
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))
                if not early_stop_count:
                    self.load_model()
                    print('result preview...')
                    file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'bcq_eval_log%s.json'%cfg.seed))
                    logging.getLogger('').addHandler(file_handler)
                    logging.info(str(cfg))

                    return
        self.load_model()
        print('result preview...')
        file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'bcq_eval_log%s.json'%cfg.seed))
        logging.getLogger('').addHandler(file_handler)
        logging.info(str(cfg))



    def validate(self, replay_buffer=None):
        print("************* validate test **********")
        #pdb.set_trace()
        self.m.eval()
        loss_log = {'total':0, 'vae': 0, 'actor':0, 'critic': 0}
        cnt = 0
        for iter_num in range(replay_buffer.val_size // cfg.batch_size):
            bgt = time.time()
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_not_dones = replay_buffer.sample(cfg.batch_size, data_type='train')
            inputs = {}
            batch_states = self.tokenize_sentence(batch_states, 'bspn')
            batch_actions = self.tokenize_sentence(batch_actions, 'aspn')
            batch_next_states = self.tokenize_sentence(batch_next_states, 'bspn')
            #TODO: see if set unknown matters here

            inputs['state_np'] = utils.padSeqs(batch_states, truncated=cfg.truncated, trunc_method='pre') 
            inputs['state_unk_np'] = deepcopy(inputs['state_np'])
            inputs['state_unk_np'][inputs['state_unk_np']>=self.reader.vocab_size] = 2   # <unk>
            inputs['action_np'] = utils.padSeqs(batch_actions, truncated=cfg.truncated, trunc_method='pre') 
            inputs['action_unk_np'] = deepcopy(inputs['action_np'])
            inputs['action_unk_np'][inputs['action_unk_np']>=self.reader.vocab_size] = 2   # <unk>
            inputs['next_state_np'] = utils.padSeqs(batch_next_states, truncated=cfg.truncated, trunc_method='pre') 
            inputs['next_state_unk_np'] = deepcopy(inputs['next_state_np'])
            inputs['next_state_unk_np'][inputs['next_state_unk_np']>=self.reader.vocab_size] = 2   # <unk>
            inputs['reward_unk_np'] = batch_rewards
            inputs['not_done_unk_np'] = batch_not_dones

            inputs = self.add_torch_input(inputs)

            vae_loss, actor_loss, critic_loss = self.m.valid_forward(inputs['state'], inputs['action'], inputs['next_state'], inputs['reward'], inputs['not_done'], batch_size=cfg.batch_size)
            total_loss = vae_loss + actor_loss + critic_loss
            loss_log['total'] += float(total_loss)
            loss_log['vae'] += float(vae_loss)
            loss_log['actor'] += float(actor_loss)
            loss_log['critic'] += float(critic_loss)
            cnt += 1

        logging.info('validation vae: %2.1f  actor: %2.1f  critic: %2.1f'%(loss_log['vae']/cnt, loss_log['actor']/cnt, loss_log['critic']/cnt))
        self.m.train()
        return loss_log['total']/cnt, loss_log['vae']/cnt, loss_log['actor']/cnt, loss_log['critic']/cnt

        

    def save_model(self, epoch, path=None, critical=False):
        if not cfg.save_log:
            return
        if not path:
            path = cfg.bcq_model_path
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)
        logging.info('BCQ Model saved {}'.format(path))

    def load_model(self, path=None):
        if not path:
            path = cfg.bcq_model_path
        if cfg.cuda:
            device = torch.device(f"cuda:{cfg.cuda_device[0]}") if len(cfg.cuda_device) == 1 else torch.device("cuda")
        else:
            device = torch.device("cpu")
        #all_state = torch.load(path, map_location='cpu')
        all_state = torch.load(path, map_location=device)
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)
        logging.info('BCQ Model loaded')

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        if not cfg.multi_gpu:
            initial_arr = self.m.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.embedding.weight.data.copy_(emb)
        else:
            initial_arr = self.m.module.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.module.embedding.weight.data.copy_(emb)


    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        print('total trainable params: %d' % param_cnt)
        return param_cnt

def parse_arg_cfg(args):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            if k in other_config.keys():
                if isinstance(other_config[k],bool):
                    other_config[k] = eval(v)
                elif isinstance(other_config[k],int):
                    other_config[k] = int(v)
                elif isinstance(other_config[k],float):
                    other_config[k] = float(v)
                elif isinstance(other_config[k],str):
                    other_config[k] = str(v)
                elif other_config[k] is None:
                    other_config[k] = str(v)
                else:
                    raise Exception('Unkown config type:{}'.format(k))
                continue
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    if 'auto' in v:
                        v = [int(get_freer_device())]
                    else:
                        v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return

def get_freer_device():
    nvidia_smi.nvmlInit()
    gpu_free = []
    for gpu_id in range(nvidia_smi.nvmlDeviceGetCount()):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_free.append(mem.free)
    return np.argmax(gpu_free)


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    cfg.mode = args.mode
    if args.mode == 'test' or args.mode=='adjust':
        parse_arg_cfg(args)
        cfg_load = json.loads(open(os.path.join(cfg.eval_load_path, 'config.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_load_path', 'eval_per_domain', 'use_true_pv_resp',
                        'use_true_prev_bspn','use_true_prev_aspn','use_true_curr_bspn','use_true_curr_aspn',
                        'name_slot_unable', 'book_slot_unable','count_req_dials_only','log_time', 'model_path',
                        'result_path', 'model_parameters', 'multi_gpu', 'use_true_bspn_for_ctr_eval', 'nbest',
                        'limit_bspn_vocab', 'limit_aspn_vocab', 'same_eval_as_cambridge', 'beam_width',
                        'use_true_domain_for_ctr_eval', 'use_true_prev_dspn', 'aspn_decode_mode',
                        'beam_diverse_param', 'same_eval_act_f1_as_hdsa', 'topk_num', 'nucleur_p',
                        'act_selection_scheme', 'beam_penalty_type', 'record_mode' , 'data_file', 'use_true_db_pointer']:
                continue
            setattr(cfg, k, v)
            cfg.model_path = os.path.join(cfg.eval_load_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.eval_load_path, 'result.csv')
            
        other_config_path = os.path.join(cfg.eval_load_path, 'other_config.json')
        if os.path.isfile(other_config_path):
            other_cfg_load = json.loads(open(other_config_path, 'r').read())
            for k,v in other_cfg_load.items():
                other_config[k]=v
    else:
        parse_arg_cfg(args)
        
        print(other_config)
        hasher = hashlib.sha1(str(other_config).encode('utf-8'))
        hash_ = base64.urlsafe_b64encode(hasher.digest()[:10])
        hash_ = re.sub(r'[^A-Za-z]', '', str(hash_))
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        
        if cfg.exp_path in ['' , 'to be generated']:
            cfg.exp_path = 'experiments/{}_{}_{}_sd{}_lr{}_bs{}_sp{}_dc{}_act{}_0.05_hash{}/'.format(current_date, '-'.join(cfg.exp_domains),
                                                                                            cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                                            cfg.early_stop_count, cfg.weight_decay_count, cfg.enable_aspn,
                                                                                            hash_)
            if cfg.save_log:
                if os.path.exists(cfg.exp_path):
                    shutil.rmtree(cfg.exp_path)
                os.mkdir(cfg.exp_path)

            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')

            if cfg.enable_tensorboard:
                cfg.tensorboard_path = os.path.join(cfg.exp_path, 'runs')
                print('tensorboard_path {}'.format(cfg.tensorboard_path))
            cfg.eval_load_path = cfg.exp_path
        elif cfg.train_bcq:
            cfg.model_path = os.path.join(cfg.exp_path, 'bcq_model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'bcq_result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            if cfg.enable_tensorboard:
                cfg.tensorboard_path = os.path.join(cfg.exp_path, 'bcq_runs')
                print('tensorboard_path {}'.format(cfg.tensorboard_path))
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if len(cfg.cuda_device)==1:
            cfg.multi_gpu = False
            torch.cuda.set_device(cfg.cuda_device[0])
        else:
            # cfg.batch_size *= len(cfg.cuda_device)
            cfg.multi_gpu = True
            torch.cuda.set_device(cfg.cuda_device[0])
        logging.info('Device: {}'.format(torch.cuda.current_device()))


    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.train_bcq:
        m = BCQModel()
    elif cfg.use_bcq:
        bcq_m = BCQModel()
        bcq_m.load_model(cfg.bcq_model_path)
        m = Model(bcq_m)
    else:
        m = Model()
    cfg.model_parameters = m.count_params()
    logging.info(str(cfg))
    logging.info(json.dumps(other_config,indent=2))

    if cfg.train_bcq:
        if cfg.save_log:
            # open(cfg.exp_path + 'config.json', 'w').write(str(cfg))
            m.reader.vocab.save_vocab(cfg.vocab_path_eval)
            with open(os.path.join(cfg.exp_path, 'bcq_config.json'), 'w') as f:
                json.dump({**cfg.__dict__, **other_config}, f, indent=2)
                
            with open(os.path.join(cfg.exp_path, 'bcq_other_config.json'), 'w') as f:
                json.dump(other_config,f,indent=2)
        # m.load_glove_embedding()
        m.train()
    elif args.mode == 'train' and not cfg.train_bcq:
        if cfg.save_log:
            # open(cfg.exp_path + 'config.json', 'w').write(str(cfg))
            m.reader.vocab.save_vocab(cfg.vocab_path_eval)
            with open(os.path.join(cfg.exp_path, 'config.json'), 'w') as f:
                json.dump({**cfg.__dict__, **other_config}, f, indent=2)
                
            with open(os.path.join(cfg.exp_path, 'other_config.json'), 'w') as f:
                json.dump(other_config,f,indent=2)
        # m.load_glove_embedding()
        m.train()
    elif args.mode == 'adjust':
        m.load_model(cfg.model_path)
        m.train()
    elif args.mode == 'test':
        m.load_model(cfg.model_path)
        # m.train()
        m.eval(data='test')
    if cfg.enable_tensorboard:
        m.writer.close()



if __name__ == '__main__':
    main()
