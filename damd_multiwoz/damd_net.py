import copy, operator
from queue import PriorityQueue
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical
import utils
from config import global_config as cfg
from otherconfig import other_config
import pdb
from info_nce import InfoNCE, info_nce
torch.autograd.set_detect_anomaly(True)

np.set_printoptions(precision=2,suppress=True)

def cuda_(var):
    # cfg.cuda_device[0]
    return var.cuda() if cfg.cuda else var


def init_gru(gru):
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()
    gru.apply(weight_reset)
    # gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i : i + gru.hidden_size], gain=1)


def label_smoothing(labels, smoothing_rate, vocab_size_oov):
    with torch.no_grad():
        confidence = 1.0 - smoothing_rate
        low_confidence = (1.0 - confidence) / labels.new_tensor(vocab_size_oov - 1)
        y_tensor = labels.data if isinstance(labels, Variable) else labels
        y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
        n_dims = vocab_size_oov
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).fill_(low_confidence).scatter_(1, y_tensor, confidence)
        y_one_hot = cuda_(y_one_hot.view(*labels.shape, -1))
    return y_one_hot


def get_one_hot_input(x_input_np):
    """
    sparse input of
    :param x_input_np: [B, Tenc]
    :return: tensor: [B,Tenc, V+Tenc]
    """
    def to_one_hot(y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
        y_tensor = y.data if isinstance(y, Variable) else y
        y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).fill_(0.).scatter_(1, y_tensor, 1)   #1e-10
        return cuda_(y_one_hot.view(*y.shape, -1))

    new_input_np = np.copy(x_input_np)
    for b in range(x_input_np.shape[0]):
        for t in range(x_input_np.shape[1]):
            if x_input_np[b][t] == 2:
                new_input_np[b][t] = cfg.vocab_size + t
    # x_input_ = (x_input_np==unk)*(widx_offset + cfg.vocab_size-unk) + x_input_np

    # input_np[input_np==2] = 0
    input_t = cuda_(torch.from_numpy(new_input_np).type(torch.LongTensor))   #[B, T]
    input_t_onehot = to_one_hot(input_t, n_dims=cfg.vocab_size+input_t.size()[1])   #[B,T,V+T]
    input_t_onehot[:, :, 0] = 0.   #<pad> to zero
    # print(x_input_np.shape[0])
    # return torch.Tensor(x_input_np.shape[1], x_input_np.shape[0], cfg.vocab_size + x_input_np.shape[0]).fill_(1e-10)
    return input_t_onehot


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        # self.v = nn.Parameter(torch.zeros(hidden_size))
        # stdv = 1. / math.sqrt(self.v.size(0))
        # self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        :param hidden: tensor of size [n_layer, B, H]
        :param encoder_outputs: tensor of size [B,T, H]
        """
        attn_energies = self.score(hidden, encoder_outputs)   # [B,T,H]
        if mask is None:
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
        else:
            # mask = (inp_seqs > 0).float()
            attn_energies.masked_fill_(mask, -1e20)
            # print('masked attn:', attn_energies[0:2,:,:])
            normalized_energy = F.softmax(attn_energies, dim=2)  # [B,1,T]
            # print('masked normalized attn:', normalized_energy[0:2,:,:])

        context = torch.bmm(normalized_energy, encoder_outputs)  # [B,1,H]
        return context  # [B,1, H]

    def score(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)   # [B,T,H]
        energy = torch.tanh(self.attn(torch.cat([H, encoder_outputs], 2)))  # [B,T,2H]->[B,T,H]
        energy = self.v(energy).transpose(1,2)   # [B,1,T]
        return energy


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, hidden_size, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out


class MultiLayerGRUwithLN(nn.Module):
    """multi-layer GRU with layer normalization """
    def __init__(self, input_size, hidden_size, layer_num = 1, bidirec = False,
                        layer_norm = False, skip_connect = False , dropout = .0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bidirec = bidirec
        self.layer_norm = layer_norm
        self.skip_connect = skip_connect
        self.dropout = dropout
        self.model_layers = nn.ModuleDict()
        self.all_weights = []
        for l in range(self.layer_num):
            if l == 0:
                gru = nn.GRU(self.input_size, self.hidden_size, num_layers=1, dropout=self.dropout,
                                                  bidirectional=self.bidirec, batch_first=True)
            else:
                input_size = self.hidden_size if not self.bidirec else 2 * self.hidden_size
                gru = nn.GRU(input_size, self.hidden_size, num_layers=1, dropout=self.dropout,
                                        bidirectional=self.bidirec, batch_first=True)
            self.model_layers['gru_'+str(l)] = gru
            self.all_weights.extend(gru.all_weights)
            if self.layer_norm:
                output_size = self.hidden_size if not self.bidirec else 2 * self.hidden_size
                # ln = LayerNormalization(output_size)
                ln = nn.LayerNorm(output_size)
                self.model_layers['ln_'+str(l)] = ln

    def forward(self, inputs, hidden=None):
        """[summary]

        :param inputs: tensor of size [B, T, H]
        :param hidden: tensor of size [n_layer*bi-direc,B,H]
        :returns: in_l: tensor of size [B, T, H * bi-direc]
                      hs: tensor of size [n_layer * bi-direc,B,H]
        """
        batch_size = inputs.size()[0]
        in_l, last_input = inputs, None
        hs = []
        if hidden:
            hiddens = hidden.view(self.layer_num, self.bidirec, batch_size, self.hidden_size)
        for l in range(self.layer_num):
            init_hs = hiddens[l] if hidden else None
            in_l, hs_l = self.model_layers['gru_'+str(l)](in_l, init_hs)
            hs.append(hs_l)
            if self.layer_norm:
                in_l = self.model_layers['ln_'+str(l)](in_l)
            if self.dropout>0 and l < (self.layer_num - 1):
                in_l = F.dropout(in_l)
            if self.skip_connect and last_input is not None:
                in_l = in_l + last_input
            last_input = in_l
        hs = torch.cat(hs, 0)
        return in_l, hs


class biGRUencoder(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.hidden_size = cfg.hidden_size
        # self.embedding = nn.Embedding(vocab_size, embed_size)
        if cfg.enc_layer_num == 1:
            self.gru = nn.GRU(self.embed_size, cfg.hidden_size, cfg.enc_layer_num, dropout=cfg.dropout,
                                          bidirectional=True, batch_first=True)
        else:
            self.gru = MultiLayerGRUwithLN(self.embed_size, cfg.hidden_size, cfg.enc_layer_num, bidirec = True,
                        layer_norm = cfg.layer_norm, skip_connect = cfg.skip_connect, dropout = cfg.dropout)
        init_gru(self.gru)


    def forward(self, input_seqs, hidden=None):
        """
        forward procedure. No need for inputs to be sorted
        :param input_seqs: Variable of [B,T]
        :param hidden:
        :return: outputs [B,T,H], hidden [n_layer*bi-direc,B,H]
        """
        embedded = self.embedding(input_seqs)
        #self.gru.flatten_parameters()
        outputs, hidden = self.gru(embedded, hidden)
        # print(outputs.size())
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Copy(nn.Module):
    def __init__(self, hidden_size, copy_weight=1.):
        super().__init__()
        self.Wcopy = nn.Linear(hidden_size, hidden_size)
        self.copy_weight = copy_weight


    def forward(self, enc_out_hs, dec_hs):
        """
        get unnormalized copy score
        :param enc_out_hs: [B, Tenc,  H]
        :param dec_hs: [B, Tdec, H]   testing: Tdec=1
        :return: raw_cp_score of each position, size [B, Tdec, Tenc]
        """
        # print(B,H,Tdec, enc_out_hs.size(0))
        raw_cp_score = torch.tanh(self.Wcopy(enc_out_hs))   #[B,Tenc,H]
        raw_cp_score = torch.einsum('beh,bdh->bde',raw_cp_score, dec_hs)    #[B, Tdec, Tenc]
        return raw_cp_score * self.copy_weight


# def get_final_scores(raw_scores, word_onehot_input, input_idx_oov, vocab_size_oov):
#     """
#     :param raw_scores: list of tensor of size [B, Tdec, V], [B, Tdec, Tenc1], [B, Tdec, Tenc1] ...
#     :param word_onehot_input: list of nparray of size [B, Tenci, V+Tenci]
#     :param input_idx_oov: list of nparray of size [B, Tenc]
#     :param vocab_size_oov:
#     :returns: tensor of size [B, Tdec, vocab_size_oov]
#     """

#     cum_idx = [score.size(2) for score in raw_scores]
#     for i in range(len(cum_idx) - 1):
#         cum_idx[i + 1] += cum_idx[i]
#     cum_idx.insert(0, 0)
#     logsoftmax = torch.nn.LogSoftmax(dim=2)
#     normalized_scores = logsoftmax(torch.cat(raw_scores, dim=2))   #[B,Tdec, V+Tenc1+Tenc2+...]
#     normalized_scores.size()

#     # print('normalized_gen_scores:' , normalized_scores.cpu().detach().numpy()[0,:5, 0:40])


#     gen_score = normalized_scores[:, :, cum_idx[0]:cum_idx[1]]   # [B, Tdec, V]
#     Tdec = gen_score.size(1)
#     B = gen_score.size(0)
#     V = gen_score.size(2)

#     total_score = cuda_(torch.zeros(B, Tdec, vocab_size_oov)).fill_(-1e20)   # [B, Tdec, vocab_size_oov]
#     c_to_g_scores = []
#     for i in range(1, len(cum_idx) - 1):
#         cps = normalized_scores[:, :, cum_idx[i]:cum_idx[i+1]]   #[B, Tdec, Tenc_i]
#         # print('normalized_cp_scores:' , cps.cpu().detach().numpy()[0,:5, 0:40])
#         one_hot = word_onehot_input[i-1]   #[B, Tenc_i, V+Tenc_i]
#         cps = torch.einsum('imj,ijn->imn', cps, one_hot)   #[B, Tdec, V+Tenc_i]
#         cps[cps==0] = -1e20   # zero prob -> -inf log prob
#         c_to_g_scores.append(cps[:, :, :V])
#         cp_score = cps[:, :, V:]
#         # avail_copy_idx = np.argwhere(input_idx_oov[i-1]>V)
#         avail_copy_idx = (input_idx_oov[i-1]>V).nonzero()
#         # print(len(copy_idx))
#         for idx in avail_copy_idx:
#             b, t = idx[0], idx[1]
#             ts = total_score[b, :, input_idx_oov[i-1][b, t]].view(Tdec, 1)
#             cs = cp_score[b, :, t].view(Tdec, 1)
#             total_score[b, :, input_idx_oov[i-1][b, t]] = torch.logsumexp(torch.cat([ts, cs], 0), 0)

#     m = torch.stack([gen_score] + c_to_g_scores, 3)
#     # print(m[0, :30, :])
#     gen_score = torch.logsumexp(m, 3)
#     total_score[:, :, :V] = gen_score
#     # print('total_score:' , total_score.cpu().detach().numpy()[0,:3, 0:40])
#     return total_score.contiguous()   #[B, Tdec, vocab_size_oov]

def get_final_scores(raw_scores, word_onehot_input, input_idx_oov, vocab_size_oov):
    """
    :param raw_scores: list of tensor of size [B, Tdec, V], [B, Tdec, Tenc1], [B, Tdec, Tenc1] ...
    :param word_onehot_input: list of nparray of size [B, Tenci, V+Tenci]
    :param input_idx_oov: list of nparray of size [B, Tenc]
    :param vocab_size_oov:
    :returns: tensor of size [B, Tdec, vocab_size_oov]
    """


    for idx, raw_sc in enumerate(raw_scores):
        if idx==0: continue
        one_hot = word_onehot_input[idx-1]   #[B, Tenc_i, V+Tenc_i]
        cps = torch.einsum('imj,ijn->imn', raw_sc, one_hot)   #[B, Tdec, V+Tenc_i]
        # cps[cps==0] = -1e20   # zero prob -> -inf log prob
        raw_scores[idx] = cps

    cum_idx = [score.size(2) for score in raw_scores]
    for i in range(len(cum_idx) - 1):
        cum_idx[i + 1] += cum_idx[i]
    cum_idx.insert(0, 0)

    logsoftmax = torch.nn.LogSoftmax(dim=2)
    normalized_scores = logsoftmax(torch.cat(raw_scores, dim=2))   #[B,Tdec, V+V+Tenc1+V+Tenc2+...]
    # print(normalized_scores.size())

    # print('normalized_gen_scores:' , normalized_scores.cpu().detach().numpy()[0,:5, 0:40])


    gen_score = normalized_scores[:, :, cum_idx[0]:cum_idx[1]]   # [B, Tdec, V]
    Tdec = gen_score.size(1)
    B = gen_score.size(0)
    V = gen_score.size(2)

    total_score = cuda_(torch.zeros(B, Tdec, vocab_size_oov)).fill_(-1e20)   # [B, Tdec, vocab_size_oov]
    c_to_g_scores = []
    for i in range(1, len(cum_idx) - 1):
        cps = normalized_scores[:, :, cum_idx[i]:cum_idx[i+1]]   #[B, Tdec, V+Tenc_i]
        # print('normalized_cp_scores:' , cps.cpu().detach().numpy()[0,:5, 0:40])
        c_to_g_scores.append(cps[:, :, :V])
        cp_score = cps[:, :, V:]
        avail_copy_idx = (input_idx_oov[i-1]>=V).nonzero()
        # print(len(copy_idx))
        for idx in avail_copy_idx:
            b, t = idx[0], idx[1]
            ts = total_score[b, :, input_idx_oov[i-1][b, t]].view(Tdec,1)
            cs = cp_score[b, :, t].view(Tdec,1)
            total_score[b, :, input_idx_oov[i-1][b, t]] = torch.logsumexp(torch.cat([ts, cs], 1), 1)

    gen_score = torch.logsumexp(torch.stack([gen_score] + c_to_g_scores, 3), 3)
    total_score[:, :, :V] = gen_score
    # print('total_score:' , total_score.cpu().detach().numpy()[0,:3, 0:40])
    return total_score.contiguous()   #[B, Tdec, vocab_size_oov]


class DomainSpanDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, Wgen=None, dropout=0.):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        self.gru = nn.GRU(3*cfg.hidden_size + self.embed_size, cfg.hidden_size, cfg.dec_layer_num,
                                     dropout=cfg.dropout, batch_first=True)
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen

        self.attn_user = Attn(cfg.hidden_size)
        self.attn_pvresp = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)
        self.attn_pvdspn = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_pvdspn = Copy(cfg.hidden_size)

    def forward(self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, mode='train'):
        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        # embed_last_w = self.dropout_layer(embed_last_w)
        gru_input.append(embed_last_w)
        # print(embed_last_w.size())

        if first_step:
            self.mask_user = (inputs['user']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvresp = (inputs['pv_resp']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvdspn = (inputs['pv_dspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            # print('masks:', self.mask_user.device, self.mask_pvresp.device, self.mask_pvbspn.device)
        if mode == 'test' and not first_step:
            self.mask_pvresp = (inputs['pv_resp']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvdspn = (inputs['pv_dspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]

        # print('user:', inputs['user'][0:2, :])
        context_user = self.attn_user(dec_last_h, hidden_states['user'], self.mask_user)
        # context_user = self.attn_user(dec_last_h, huser, self.mask_user)
        gru_input.append(context_user)
        # print(context_user.size())
        if not first_turn:
            context_pvresp = self.attn_pvresp(dec_last_h, hidden_states['resp'], self.mask_pvresp)
            context_pvdspn = self.attn_pvdspn(dec_last_h, hidden_states['dspn'], self.mask_pvdspn)
        else:
            batch_size = inputs['user'].size(0)
            context_pvresp = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))#.to(context_user.device)
            context_pvdspn = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))#.to(context_user.device)
        gru_input.append(context_pvresp)
        gru_input.append(context_pvdspn)
        # print(context_pvbspn.size())

        #self.gru.flatten_parameters()
        gru_out, dec_last_h = self.gru(torch.cat(gru_input, 2), dec_last_h)   # [B, 1, H], [n_layer, B, H]
        # gru_out = self.dropout_layer(gru_out)
        # print(gru_out.size())
        return dec_last_h


    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False):
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)    #[B, Tdec, H]
        raw_scores.append(raw_gen_score)

        if not first_turn:
            raw_cp_score_dspn = self.cp_pvdspn(hidden_states['dspn'], dec_hs)   #[B,Ta]
            raw_cp_score_dspn.masked_fill_(self.mask_pvdspn.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_dspn)
            word_onehot_input.append(inputs['pv_dspn_onehot'])
            input_idx_oov.append(inputs['pv_dspn_nounk'])

        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)

        return probs

class BeliefSpanDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, bspn_mode, Wgen=None, dropout=0.):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        self.bspn_mode = bspn_mode

        self.gru = nn.GRU(3*cfg.hidden_size + self.embed_size, cfg.hidden_size, cfg.dec_layer_num,
                                     dropout=cfg.dropout, batch_first=True)
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen

        self.attn_user = Attn(cfg.hidden_size)
        self.attn_pvresp = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)
        self.attn_pvbspn = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_user = Copy(cfg.hidden_size, 1.)
        self.cp_pvresp = self.cp_user if cfg.copy_param_share else Copy(cfg.hidden_size)
        self.cp_pvbspn = self.cp_user if cfg.copy_param_share else Copy(cfg.hidden_size, 1.)

        self.mask_user = None
        self.mask_pvresp = None
        self.mask_pvbspn = None

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)  # input dropout



    def forward(self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, mode='train'):
    # def forward(self, inputs, huser, hresp, hbspn, dec_last_w, dec_last_h, first_turn, first_step):
        """[summary]
        :param inputs: inputs dict
        :param hidden_states: hidden states dict, size [B, T, H]
        :param dec_last_w: word index of last decoding step
        :param dec_last_h: hidden state of last decoding step
        :param first_turn: [description], defaults to False
        :returns: [description]
        """

        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        # embed_last_w = self.dropout_layer(embed_last_w)
        gru_input.append(embed_last_w)
        # print(embed_last_w.size())

        if first_step:
            self.mask_user = (inputs['user']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvresp = (inputs['pv_resp']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvbspn = (inputs['pv_'+self.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            # print('masks:', self.mask_user.device, self.mask_pvresp.device, self.mask_pvbspn.device)
        if mode == 'test' and not first_step:
            self.mask_pvresp = (inputs['pv_resp']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvbspn = (inputs['pv_'+self.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]

        # print('user:', inputs['user'][0:2, :])
        context_user = self.attn_user(dec_last_h, hidden_states['user'], self.mask_user)
        # context_user = self.attn_user(dec_last_h, huser, self.mask_user)
        gru_input.append(context_user)
        # print(context_user.size())
        if not first_turn:
            context_pvresp = self.attn_pvresp(dec_last_h, hidden_states['resp'], self.mask_pvresp)
            context_pvbspn = self.attn_pvbspn(dec_last_h, hidden_states[self.bspn_mode], self.mask_pvbspn)

            # context_pvresp = self.attn_pvresp(dec_last_h, hresp, self.mask_pvresp)
            # context_pvbspn = self.attn_pvbspn(dec_last_h, hbspn, self.mask_pvbspn)
        else:
            batch_size = inputs['user'].size(0)
            context_pvresp = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))#.to(context_user.device)
            context_pvbspn = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))#.to(context_user.device)
        gru_input.append(context_pvresp)
        gru_input.append(context_pvbspn)
        # print(context_pvbspn.size())

        #self.gru.flatten_parameters()
        gru_out, dec_last_h = self.gru(torch.cat(gru_input, 2), dec_last_h)   # [B, 1, H], [n_layer, B, H]
        # gru_out = self.dropout_layer(gru_out)
        # print(gru_out.size())
        return dec_last_h


    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False):
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)    #[B, Tdec, V]
        raw_scores.append(raw_gen_score)

        raw_cp_score_user = self.cp_user(hidden_states['user'], dec_hs)   #[B, Tdec,Tu]
        raw_cp_score_user.masked_fill_(self.mask_user.repeat(1,Tdec,1), -1e20)
        raw_scores.append(raw_cp_score_user)
        word_onehot_input.append(inputs['user_onehot'])
        input_idx_oov.append(inputs['user_nounk'])

        if not first_turn:
            raw_cp_score_pvresp = self.cp_pvresp(hidden_states['resp'], dec_hs)   #[B, Tdec,Tr]
            raw_cp_score_pvresp.masked_fill_(self.mask_pvresp.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_pvresp)
            word_onehot_input.append(inputs['pv_resp_onehot'])
            input_idx_oov.append(inputs['pv_resp_nounk'])

            raw_cp_score_pvbspn = self.cp_pvbspn(hidden_states[self.bspn_mode], dec_hs)   #[B, Tdec, Tb]
            raw_cp_score_pvbspn.masked_fill_(self.mask_pvbspn.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_pvbspn)
            word_onehot_input.append(inputs['pv_%s_onehot'%self.bspn_mode])
            input_idx_oov.append(inputs['pv_%s_nounk'%self.bspn_mode])

        # print('bspn:' , inputs['bspn'][0, 0:10])
        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)   # [B, V_oov]

        return probs

class CntfactBeliefSpanDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, cntfact_bspn_mode, Wgen=None, dropout=0.):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        self.cntfact_bspn_mode = cntfact_bspn_mode
        self.bspn_mode = cfg.bspn_mode #To use in contrast learning

        self.gru = nn.GRU(3*cfg.hidden_size + self.embed_size, cfg.hidden_size, cfg.dec_layer_num,
                                     dropout=cfg.dropout, batch_first=True)
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen

        self.attn_user = Attn(cfg.hidden_size)
        self.attn_pvresp = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)
        self.attn_pvbspn = self.attn_user if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_user = Copy(cfg.hidden_size, 1.)
        self.cp_pvresp = self.cp_user if cfg.copy_param_share else Copy(cfg.hidden_size)
        self.cp_pvbspn = self.cp_user if cfg.copy_param_share else Copy(cfg.hidden_size, 1.)

        self.mask_user = None
        self.mask_pvresp = None
        self.mask_pvbspn = None

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)  # input dropout



    def forward(self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, mode='train', return_bspn=False, idx=None):
    # def forward(self, inputs, huser, hresp, hbspn, dec_last_w, dec_last_h, first_turn, first_step):
        """[summary]
        :param inputs: inputs dict
        :param hidden_states: hidden states dict, size [B, T, H]
        :param dec_last_w: word index of last decoding step
        :param dec_last_h: hidden state of last decoding step
        :param first_turn: [description], defaults to False
        :param return_bspn: if true, return real belief's hidden state for contrast learning
        :param idx: idx of multiple cntfactbspn
        :returns: [description]
        """
        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        # embed_last_w = self.dropout_layer(embed_last_w)
        gru_input.append(embed_last_w)
        # print(embed_last_w.size())

        if first_step:
            self.mask_user = (inputs['user']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvresp = (inputs['pv_resp']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if return_bspn: 
                self.mask_pvbspn = (inputs['pv_'+self.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            else: 
                if idx is None:
                # normal cntfact process
                    self.mask_pvbspn = (inputs['pv_'+self.cntfact_bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
                else:
                # multi cntfact process
                    self.mask_pvbspn = (inputs['pv_'+self.cntfact_bspn_mode][idx]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            # print('masks:', self.mask_user.device, self.mask_pvresp.device, self.mask_pvbspn.device)
        if mode == 'test' and not first_step:
            self.mask_pvresp = (inputs['pv_resp']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if return_bspn:
                self.mask_pvbspn = (inputs['pv_'+self.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            else:
                if idx is None:
                    self.mask_pvbspn = (inputs['pv_'+self.cntfact_bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
                else:
                    self.mask_pvbspn = (inputs['pv_'+self.cntfact_bspn_mode][idx]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
        # print('user:', inputs['user'][0:2, :])
        context_user = self.attn_user(dec_last_h, hidden_states['user'], self.mask_user)
        # context_user = self.attn_user(dec_last_h, huser, self.mask_user)
        gru_input.append(context_user)
        # print(context_user.size())
        if not first_turn:
            context_pvresp = self.attn_pvresp(dec_last_h, hidden_states['resp'], self.mask_pvresp)
            if return_bspn:
                context_pvbspn = self.attn_pvbspn(dec_last_h, hidden_states[self.bspn_mode], self.mask_pvbspn)
            else:
                if idx is None:
                    context_pvbspn = self.attn_pvbspn(dec_last_h, hidden_states[self.cntfact_bspn_mode], self.mask_pvbspn)
                else:
                    context_pvbspn = self.attn_pvbspn(dec_last_h, hidden_states[self.cntfact_bspn_mode][idx], self.mask_pvbspn)
                

            # context_pvresp = self.attn_pvresp(dec_last_h, hresp, self.mask_pvresp)
            # context_pvbspn = self.attn_pvbspn(dec_last_h, hbspn, self.mask_pvbspn)
        else:
            batch_size = inputs['user'].size(0)
            context_pvresp = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))#.to(context_user.device)
            context_pvbspn = cuda_(torch.zeros(batch_size, 1, cfg.hidden_size))#.to(context_user.device)
        gru_input.append(context_pvresp)
        gru_input.append(context_pvbspn)
        # print(context_pvbspn.size())

        #self.gru.flatten_parameters()
        gru_out, dec_last_h = self.gru(torch.cat(gru_input, 2), dec_last_h)   # [B, 1, H], [n_layer, B, H]
        # gru_out = self.dropout_layer(gru_out)
        # print(gru_out.size())
        return dec_last_h


    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False, return_bspn=False, idx=None):
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)    #[B, Tdec, V]
        raw_scores.append(raw_gen_score)

        raw_cp_score_user = self.cp_user(hidden_states['user'], dec_hs)   #[B, Tdec,Tu]
        raw_cp_score_user.masked_fill_(self.mask_user.repeat(1,Tdec,1), -1e20)
        raw_scores.append(raw_cp_score_user)
        word_onehot_input.append(inputs['user_onehot'])
        input_idx_oov.append(inputs['user_nounk'])

        if not first_turn:
            raw_cp_score_pvresp = self.cp_pvresp(hidden_states['resp'], dec_hs)   #[B, Tdec,Tr]
            raw_cp_score_pvresp.masked_fill_(self.mask_pvresp.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_pvresp)
            word_onehot_input.append(inputs['pv_resp_onehot'])
            input_idx_oov.append(inputs['pv_resp_nounk'])
            if return_bspn:
                raw_cp_score_pvbspn = self.cp_pvbspn(hidden_states[self.bspn_mode], dec_hs)   #[B, Tdec, Tb]
                raw_cp_score_pvbspn.masked_fill_(self.mask_pvbspn.repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_pvbspn)
                word_onehot_input.append(inputs['pv_%s_onehot'%self.bspn_mode])
                input_idx_oov.append(inputs['pv_%s_nounk'%self.bspn_mode])
            else:
                raw_cp_score_pvbspn = self.cp_pvbspn(hidden_states[self.cntfact_bspn_mode], dec_hs)   #[B, Tdec, Tb]
                raw_cp_score_pvbspn.masked_fill_(self.mask_pvbspn.repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_pvbspn)
                word_onehot_input.append(inputs['pv_%s_onehot'%self.cntfact_bspn_mode])
                input_idx_oov.append(inputs['pv_%s_nounk'%self.cntfact_bspn_mode])

        # print('bspn:' , inputs['bspn'][0, 0:10])
        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)   # [B, V_oov]

        return probs

class ActSpanDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, Wgen = None, dropout=0.):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        input_dim = cfg.hidden_size + self.embed_size + cfg.pointer_dim
        if cfg.use_pvaspn:
            input_dim += cfg.hidden_size
        if cfg.enable_bspn:
            input_dim += cfg.hidden_size
        if cfg.enable_dspn :
            input_dim += cfg.hidden_size
        if cfg.use_bcq:
            input_dim += cfg.hidden_size

        self.gru = nn.GRU(input_dim, cfg.hidden_size, cfg.dec_layer_num,
                                    dropout=cfg.dropout, batch_first=True)
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen

        self.attn_usdx = Attn(cfg.hidden_size)
        if cfg.enable_bspn:
            self.attn_bspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)
        if cfg.enable_dspn:
            self.attn_dspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)
        self.attn_pvaspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_pvaspn = Copy(cfg.hidden_size)
        self.cp_dspn = self.cp_pvaspn if cfg.copy_param_share else Copy(cfg.hidden_size)
        self.cp_bspn = self.cp_pvaspn if cfg.copy_param_share else Copy(cfg.hidden_size)

        self.mask_usdx = None
        self.mask_bspn = None
        self.mask_dspn = None
        self.mask_pvaspn = None

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(cfg.dropout)  # input dropout


    def forward(self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, bidx = None, mode='train'):
    # def forward(self, inputs, husdx, hbspn, haspn, dec_last_w, dec_last_h, first_turn, first_step):

        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        # embed_last_w = self.dropout_layer(embed_last_w)
        gru_input.append(embed_last_w)

        if first_step:
            self.mask_usdx = (inputs['usdx']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            self.mask_pvaspn = (inputs['pv_aspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if cfg.enable_cntfact_reward:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_bspn and not cfg.enable_cntfact:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_cntfact and not cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.cntfact_bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)
            else:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)
            if cfg.enable_dspn: 
                self.mask_dspn = (inputs['dspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
        if mode == 'test' and not first_step:
            self.mask_pvaspn = (inputs['pv_aspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if cfg.enable_cntfact_reward:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_bspn and not cfg.enable_cntfact:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_cntfact and not cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.cntfact_bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            else:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if cfg.enable_dspn:
                self.mask_dspn = (inputs['dspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]

        if bidx is None:
            context_usdx = self.attn_usdx(dec_last_h, hidden_states['usdx'], self.mask_usdx)
        else:
            context_usdx = self.attn_usdx(dec_last_h, hidden_states['usdx'][bidx], self.mask_usdx[bidx])
        # context_usdx = self.attn_usdx(dec_last_h, husdx, self.mask_usdx)
        gru_input.append(context_usdx)
        if cfg.enable_bspn:
            if bidx is None:
                if not cfg.enable_cntfact or cfg.enable_contrast or cfg.enable_cntfact_reward:
                    context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.bspn_mode], self.mask_bspn)
                else:
                    context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.cntfact_bspn_mode], self.mask_bspn)
            else:
                if not cfg.enable_cntfact or cfg.enable_contrast:
                    context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.bspn_mode][bidx], self.mask_bspn[bidx])
                else:
                    context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.cntfact_bspn_mode][bidx], self.mask_bspn[bidx])
            gru_input.append(context_bspn)
        if cfg.enable_dspn:
            if bidx is None:
                context_dspn = self.attn_dspn(dec_last_h, hidden_states['dspn'], self.mask_dspn)
            else:
                context_dspn = self.attn_dspn(dec_last_h, hidden_states['dspn'][bidx], self.mask_dspn[bidx])
            gru_input.append(context_dspn)
        if cfg.use_pvaspn:
            if not first_turn:
                if bidx is None:
                    context_pvaspn = self.attn_pvaspn(dec_last_h, hidden_states['aspn'], self.mask_pvaspn)
                else:
                    context_pvaspn = self.attn_pvaspn(dec_last_h, hidden_states['aspn'][bidx], self.mask_pvaspn[bidx])
                # context_pvaspn = self.attn_pvaspn(dec_last_h, haspn, self.mask_pvaspn)
            else:
                if bidx is None:
                    context_pvaspn = cuda_(torch.zeros(inputs['user'].size(0), 1, cfg.hidden_size))
                else:
                    context_pvaspn = cuda_(torch.zeros(1, 1, cfg.hidden_size))
            gru_input.append(context_pvaspn)

        if bidx is None:
            gru_input.append(inputs['db'].unsqueeze(1))
        else:
            gru_input.append(inputs['db'][bidx].unsqueeze(1))
        
        if cfg.use_bcq:
            gru_input.append(inputs['bcq_action'].unsqueeze(1))

        #self.gru.flatten_parameters()
        gru_out, dec_last_h = self.gru(torch.cat(gru_input, 2), dec_last_h)   # [B, 1, H], [n_layer, B, H]
        # gru_out should be the same with last_h in for 1-layer GRU decoder
        # gru_out = self.dropout_layer(gru_out)
        return dec_last_h


    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False, bidx = None):
        """[summary]
        :param dec_hs: [B, Tdec, H]
        :param dec_ws: word index [B, Tdec]
        :param dec_hs: decoder hidden states [B, Tdec, H]
        :returns: [description]
        """
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)    #[B, Tdec, H]
        raw_scores.append(raw_gen_score)
        # print('raw_gen_score:' , raw_gen_score.cpu().detach().numpy()[0,:3, 0:40])

        if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
            if bidx is None:
                raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode], dec_hs)   #[B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.bspn_mode + '_onehot'])
                input_idx_oov.append(inputs[cfg.bspn_mode + '_nounk'])
            else:
                raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode][bidx], dec_hs)   #[B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn[bidx].repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.bspn_mode + '_onehot'][bidx])
                input_idx_oov.append(inputs[cfg.bspn_mode + '_nounk'][bidx])
            # print('raw_cp_score_bspn:' , raw_cp_score_bspn.cpu().detach().numpy()[0,:3, 0:40])

        elif cfg.enable_cntfact and not cfg.enable_contrast:
            if bidx is None:
                raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.cntfact_bspn_mode], dec_hs)   #[B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.cntfact_bspn_mode + '_onehot'])
                input_idx_oov.append(inputs[cfg.cntfact_bspn_mode + '_nounk'])
            else:
                raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.cntfact_bspn_mode][bidx], dec_hs)   #[B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn[bidx].repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.cntfact_bspn_mode + '_onehot'][bidx])
                input_idx_oov.append(inputs[cfg.cntfact_bspn_mode + '_nounk'][bidx])
        
        elif cfg.enable_contrast:
            if bidx is None:
                raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode], dec_hs)   #[B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.bspn_mode + '_onehot'])
                input_idx_oov.append(inputs[cfg.bspn_mode + '_nounk'])
            else:
                raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode][bidx], dec_hs)   #[B,Tb]
                raw_cp_score_bspn.masked_fill_(self.mask_bspn[bidx].repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_bspn)
                word_onehot_input.append(inputs[cfg.bspn_mode + '_onehot'][bidx])
                input_idx_oov.append(inputs[cfg.bspn_mode + '_nounk'][bidx])


        if cfg.enable_dspn:
            if bidx is None:
                raw_cp_score_dspn = self.cp_dspn(hidden_states['dspn'], dec_hs)   #[B,Tb]
                raw_cp_score_dspn.masked_fill_(self.mask_dspn.repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_dspn)
                word_onehot_input.append(inputs['dspn_onehot'])
                input_idx_oov.append(inputs['dspn_nounk'])
            else:
                raw_cp_score_dspn = self.cp_dspn(hidden_states['dspn'][bidx], dec_hs)   #[B,Tb]
                raw_cp_score_dspn.masked_fill_(self.mask_dspn[bidx].repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_dspn)
                word_onehot_input.append(inputs['dspn_onehot'][bidx])
                input_idx_oov.append(inputs['dspn_nounk'][bidx])

        if not first_turn and cfg.use_pvaspn:
            if bidx is None:
                raw_cp_score_aspn = self.cp_pvaspn(hidden_states['aspn'], dec_hs)   #[B,Ta]
                raw_cp_score_aspn.masked_fill_(self.mask_pvaspn.repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_aspn)
                word_onehot_input.append(inputs['pv_aspn_onehot'])
                input_idx_oov.append(inputs['pv_aspn_nounk'])
            else:
                raw_cp_score_aspn = self.cp_pvaspn(hidden_states['aspn'][bidx], dec_hs)   #[B,Ta]
                raw_cp_score_aspn.masked_fill_(self.mask_pvaspn[bidx].repeat(1,Tdec,1), -1e20)
                raw_scores.append(raw_cp_score_aspn)
                word_onehot_input.append(inputs['pv_aspn_onehot'][bidx])
                input_idx_oov.append(inputs['pv_aspn_nounk'][bidx])
            # print('raw_cp_score_aspn:' , raw_cp_score_aspn.cpu().detach().numpy()[0,:3, 0:40])

        # print('aspn:' , inputs['aspn'][0, 0:3])
        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)

        return probs



class ResponseDecoder(nn.Module):
    def __init__(self, embedding, vocab_size_oov, Wgen = None, dropout=0.):
        super().__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.vsize_oov = vocab_size_oov

        gru_input_size = cfg.hidden_size + self.embed_size + cfg.pointer_dim
        if cfg.enable_bspn:
            gru_input_size += cfg.hidden_size
        if cfg.enable_aspn:
            gru_input_size += cfg.hidden_size

        self.gru = nn.GRU(gru_input_size , cfg.hidden_size, cfg.dec_layer_num,
                                        dropout=cfg.dropout, batch_first=True)
        init_gru(self.gru)

        self.Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if not Wgen else Wgen
        self.attn_usdx = Attn(cfg.hidden_size)
        if cfg.enable_bspn:
            self.attn_bspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)
        if cfg.enable_aspn:
            self.attn_aspn = self.attn_usdx if cfg.attn_param_share else Attn(cfg.hidden_size)

        self.cp_usdx = Copy(cfg.hidden_size)
        if cfg.enable_bspn:
            self.cp_bspn = self.cp_usdx if cfg.copy_param_share else Copy(cfg.hidden_size)
        if cfg.enable_aspn:
            self.cp_aspn = self.cp_usdx if cfg.copy_param_share else Copy(cfg.hidden_size)

        self.mask_usdx = None
        self.mask_bspn = None
        if cfg.enable_aspn:
            self.mask_aspn = None

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)  # input dropout


    def forward(self, inputs, hidden_states, dec_last_w, dec_last_h, first_turn, first_step, mode='train'):
    # def forward(self, inputs, husdx, hbspn, haspn, dec_last_w, dec_last_h, first_turn, first_step):

        gru_input = []
        embed_last_w = self.embedding(dec_last_w)
        # embed_last_w = self.dropout_layer(embed_last_w)
        gru_input.append(embed_last_w)

        if first_step:
            self.mask_usdx = (inputs['usdx']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_cntfact and not cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.cntfact_bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            else:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if cfg.enable_aspn:
                self.mask_aspn = (inputs['aspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
        if mode == 'test' and not first_step:
            if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_cntfact and not cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.cntfact_bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            elif cfg.enable_contrast:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            else:
                self.mask_bspn = (inputs[cfg.bspn_mode]==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]
            if cfg.enable_aspn:
                self.mask_aspn = (inputs['aspn']==0).unsqueeze(1)#.to(dec_last_w.device)     # [B,1,T]

        context_usdx = self.attn_usdx(dec_last_h, hidden_states['usdx'], self.mask_usdx)
        # context_usdx = self.attn_usdx(dec_last_h, husdx, self.mask_usdx)
        gru_input.append(context_usdx)
        if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
            context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.bspn_mode], self.mask_bspn)
            # context_bspn = self.attn_bspn(dec_last_h, hbspn, self.mask_bspn)
            gru_input.append(context_bspn)
        elif cfg.enable_cntfact and not cfg.enable_contrast:
            context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.cntfact_bspn_mode], self.mask_bspn)
            # context_bspn = self.attn_bspn(dec_last_h, hbspn, self.mask_bspn)
            gru_input.append(context_bspn)
        elif cfg.enable_contrast:
            context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.bspn_mode], self.mask_bspn)
            # context_bspn = self.attn_bspn(dec_last_h, hbspn, self.mask_bspn)
            gru_input.append(context_bspn)
        else:
            context_bspn = self.attn_bspn(dec_last_h, hidden_states[cfg.bspn_mode], self.mask_bspn)
            # context_bspn = self.attn_bspn(dec_last_h, hbspn, self.mask_bspn)
            gru_input.append(context_bspn)
        if cfg.enable_aspn:
            context_aspn = self.attn_aspn(dec_last_h, hidden_states['aspn'], self.mask_aspn)
            # context_aspn = self.attn_aspn(dec_last_h, haspn, self.mask_aspn)
            gru_input.append(context_aspn)

        gru_input.append(inputs['db'].unsqueeze(1))

        #self.gru.flatten_parameters()
        gru_out, dec_last_h = self.gru(torch.cat(gru_input, 2), dec_last_h)   # [B, 1, H], [n_layer, B, H]
        # gru_out should be the same with last_h in for 1-layer GRU decoder
        # gru_out = self.dropout_layer(gru_out)

        return dec_last_h

    def get_probs(self, inputs, hidden_states, dec_hs, first_turn=False):
        """[summary]
        :param dec_hs: [B, Tdec, H]
        :param dec_ws: word index [B, Tdec]
        :param dec_hs: decoder hidden states [B, Tdec, H]
        :returns: [description]
        """
        Tdec = dec_hs.size(1)

        raw_scores, word_onehot_input, input_idx_oov = [], [], []
        raw_gen_score = self.Wgen(dec_hs)    #[B, Tdec, H]
        raw_scores.append(raw_gen_score)
        # print('raw_gen_score:' , raw_gen_score.cpu().detach().numpy()[0,:3, 0:40])

        raw_cp_score_usdx = self.cp_usdx(hidden_states['usdx'], dec_hs)   #[B,Tu]
        raw_cp_score_usdx.masked_fill_(self.mask_usdx.repeat(1,Tdec,1), -1e20)
        raw_scores.append(raw_cp_score_usdx)
        word_onehot_input.append(inputs['usdx_onehot'])
        input_idx_oov.append(inputs['usdx_nounk'])

        if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
            raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode], dec_hs)   #[B,Tb]
            raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_bspn)
            word_onehot_input.append(inputs[cfg.bspn_mode + '_onehot'])
            input_idx_oov.append(inputs[cfg.bspn_mode + '_nounk'])
            # print('raw_cp_score_bspn:' , raw_cp_score_bspn.cpu().detach().numpy()[0,:3, 0:40])

        elif cfg.enable_cntfact and not cfg.enable_contrast:
            raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.cntfact_bspn_mode], dec_hs)   #[B,Tb]
            raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_bspn)
            word_onehot_input.append(inputs[cfg.cntfact_bspn_mode + '_onehot'])
            input_idx_oov.append(inputs[cfg.cntfact_bspn_mode + '_nounk'])
        elif cfg.enable_contrast:
            raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode], dec_hs)   #[B,Tb]
            raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_bspn)
            word_onehot_input.append(inputs[cfg.bspn_mode + '_onehot'])
            input_idx_oov.append(inputs[cfg.bspn_mode + '_nounk'])
        else:
            raw_cp_score_bspn = self.cp_bspn(hidden_states[cfg.bspn_mode], dec_hs)   #[B,Tb]
            raw_cp_score_bspn.masked_fill_(self.mask_bspn.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_bspn)
            word_onehot_input.append(inputs[cfg.bspn_mode + '_onehot'])
            input_idx_oov.append(inputs[cfg.bspn_mode + '_nounk'])

        if cfg.enable_aspn:
            raw_cp_score_aspn = self.cp_aspn(hidden_states['aspn'], dec_hs)   #[B,Ta]
            raw_cp_score_aspn.masked_fill_(self.mask_aspn.repeat(1,Tdec,1), -1e20)
            raw_scores.append(raw_cp_score_aspn)
            word_onehot_input.append(inputs['aspn_onehot'])
            input_idx_oov.append(inputs['aspn_nounk'])

        # print('resp:' , inputs['resp'][0, 0:3])
        probs = get_final_scores(raw_scores, word_onehot_input, input_idx_oov, self.vsize_oov)

        return probs


class ActSelectionModel(nn.Module):
    def __init__(self, hidden_size, length, nbest):
        super().__init__()
        self.nbest = nbest
        self.hidden_size = hidden_size
        self.length = length
        self.W1 = nn.Linear(hidden_size * length, hidden_size)
        self.W2 = nn.Linear(hidden_size, 1)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, hiddens_batch):
        """[summary]
        :param hiddens_batch: [B, nbest, T, H]
        :param decoded_batch: [B, nbest, T]
        """
        batch_size = hiddens_batch.size()[0]
        logits = hiddens_batch.view(batch_size, self.nbest, -1)
        logits = self.W2(nn.ReLU(self.W1(logits))).view(batch_size)
        logprob = self.logsoftmax(logits)   #[B,nbest]
        return logprob

class DAMD(nn.Module):
    def __init__(self, reader, bcq=None):
        super().__init__()
        self.reader = reader
        self.vocab = self.reader.vocab
        self.vocab_size = self.vocab.vocab_size
        self.vsize_oov = self.vocab.vocab_size_oov
        self.embed_size = cfg.embed_size
        self.hidden_size = cfg.hidden_size
        self.n_layer = cfg.dec_layer_num
        self.dropout = cfg.dropout
        self.max_span_len = cfg.max_span_length
        self.max_nl_len = cfg.max_nl_length
        self.teacher_force = cfg.teacher_force
        self.label_smth = cfg.label_smoothing
        self.beam_width = cfg.beam_width
        self.nbest = cfg.nbest
        if bcq is not None:
            self.bcq = bcq

        # self.module_list = nn.ModuleList()

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        # self.module_list.append(self.embedding)


        self.user_encoder = biGRUencoder(self.embedding)
        # self.module_list.append(self.user_encoder)
        if cfg.encoder_share:
            self.usdx_encoder = self.user_encoder
        else:
            self.usdx_encoder = biGRUencoder(self.embedding)
            # self.module_list.append(self.usdx_encoder)
        self.span_encoder = biGRUencoder(self.embedding)


        Wgen = nn.Linear(cfg.hidden_size, cfg.vocab_size) if cfg.copy_param_share else None

        # joint training of dialogue state tracker
        self.decoders = {}
        if cfg.enable_dspn:
            self.dspn_decoder = DomainSpanDecoder(self.embedding, self.vsize_oov, Wgen=Wgen,
                                                                                 dropout=self.dropout)
            self.decoders['dspn'] = self.dspn_decoder
        if cfg.enable_bspn and not cfg.enable_contrast:
            self.bspn_decoder = BeliefSpanDecoder(self.embedding, self.vsize_oov, cfg.bspn_mode,
                                                                             Wgen = Wgen, dropout = self.dropout)
            self.decoders[cfg.bspn_mode] = self.bspn_decoder
        if cfg.enable_cntfact and not cfg.enable_cntfact_reward:
            self.cntfact_bspn_decoder = CntfactBeliefSpanDecoder(self.embedding, self.vsize_oov, cfg.cntfact_bspn_mode,
                                                                            Wgen = Wgen, dropout = self.dropout)
            self.decoders[cfg.cntfact_bspn_mode] = self.cntfact_bspn_decoder
        if cfg.enable_contrast:
            self.bspn_decoder = CntfactBeliefSpanDecoder(self.embedding, self.vsize_oov, cfg.cntfact_bspn_mode,
                                                                            Wgen = Wgen, dropout = self.dropout)
            self.decoders[cfg.bspn_mode] = self.bspn_decoder
        if cfg.enable_aspn:
            self.aspn_decoder = ActSpanDecoder(self.embedding, self.vsize_oov,
                                                                          Wgen = Wgen, dropout = self.dropout)
            self.decoders['aspn'] = self.aspn_decoder
        self.resp_decoder = ResponseDecoder(self.embedding, self.vsize_oov,
                                                                       Wgen = Wgen, dropout = self.dropout)
        self.decoders['resp'] = self.resp_decoder

        if cfg.enable_dst and cfg.bspn_mode == 'bsdx': # False #TODO: make change if cfg.enable_dst set to True
            self.dst_decoder = BeliefSpanDecoder(self.embedding, self.vsize_oov, 'bspn',
                                                                             Wgen = Wgen, dropout = self.dropout)
            self.decoders['bspn'] = self.dst_decoder

        self.nllloss = nn.NLLLoss(ignore_index=0)
        self.nllloss_kd = nn.NLLLoss(ignore_index=0,reduction='none')

        if cfg.enable_cntfact:
            self.go_idx = {'bspn': 3, 'bsdx': 3, 'cntfact_bspn': 3, 'cntfact_bsdx': 3, 'aspn': 4, 'dspn': 9, 'resp': 1}
            self.eos_idx = {'bspn': 7, 'bsdx': 7, 'cntfact_bspn': 7, 'cntfact_bsdx': 7, 'aspn': 8, 'dspn': 10, 'resp': 6}
            self.teacher_forcing_decode = {
                'bspn': cfg.use_true_curr_bspn,
                'bsdx': cfg.use_true_curr_bspn,
                'aspn': cfg.use_true_curr_aspn,
                'cntfact_bspn': cfg.use_true_curr_cntfact_bspn,
                'cntfact_bsdx': cfg.use_true_curr_cntfact_bspn,
                'dspn': False,
                'resp': False}
            self.limited_vocab_decode = {
                'bspn': cfg.limit_bspn_vocab,
                'bsdx': cfg.limit_bspn_vocab,
                'cntfact_bspn': cfg.limit_bspn_vocab,
                'cntfact_bsdx': cfg.limit_bspn_vocab,
                'aspn': cfg.limit_aspn_vocab,
                'dspn': False,
                'resp': False}
        else:
            self.go_idx = {'bspn': 3, 'bsdx': 3, 'aspn': 4, 'dspn': 9, 'resp': 1}
            self.eos_idx = {'bspn': 7, 'bsdx': 7, 'aspn': 8, 'dspn': 10, 'resp': 6}
            self.teacher_forcing_decode = {
                'bspn': cfg.use_true_curr_bspn,
                'bsdx': cfg.use_true_curr_bspn,
                'aspn': cfg.use_true_curr_aspn,
                'dspn': False,
                'resp': False}
            self.limited_vocab_decode = {
                'bspn': cfg.limit_bspn_vocab,
                'bsdx': cfg.limit_bspn_vocab,
                'aspn': cfg.limit_aspn_vocab,
                'dspn': False,
                'resp': False}

    def supervised_loss(self, inputs, probs):
        def LabelSmoothingNLLLoss(logprob, labels):
                return -(labels * logprob).sum((1,2)).mean()
        total_loss = 0
        if cfg.enable_cntfact: #TODO: eliminate bspn and bsdx here
            losses = {'bsdx':0, 'bspn':0, 'aspn':0, 'resp':0, 'dspn':0, 'cntfact_bsdx':0, 'cntfact_bspn':0}
        else:
            losses = {'bsdx':0, 'bspn':0, 'aspn':0, 'resp':0, 'dspn':0}
        for name, prob in probs.items():
            if name == 'aspn_aug':
                continue
            # print(prob)
            # pred = torch.log(prob.view(-1, prob.size(2)))
            # print(pred[0, :50])
            if name != 'resp' or cfg.label_smoothing == .0:
                if name == 'aspn' and ('L_det' in other_config['policy_loss'] or 'L_soft' in other_config['policy_loss']):
                    pred = prob.permute(0,2,1)
                    label = inputs[name+'_4loss']

                    seq_mask = (inputs['aspn_4loss']!=0)*1                        
                    loss = self.nllloss_kd(pred, label)
                    if 'L_det' in other_config['policy_loss']:
                        loss = torch.unsqueeze(inputs['G'],dim=1) * loss * 10
                        loss = torch.sum(loss)/torch.sum(seq_mask)
                    if 'L_sto' in other_config['policy_loss']:
                        loss += self.spi_loss(inputs, probs, losses, label)
                else:
                    if name in ['pi(a|b)'] or (cfg.enable_rl and name == 'aspn'):
                        continue
                                        
                    pred = prob.view(-1, prob.size(2))   #[B,T,Voov] -> [B*T, Voov]
                    label = inputs[name+'_4loss'].view(-1)
                    # print(label[:50])
                    loss = self.nllloss(pred, label)
                total_loss += loss
                losses[name] = loss
            else:
                pred = prob.view(-1, prob.size(2))   #[B,T,Voov] -> [B*T, Voov]
                label = inputs[name+'_4loss'].view(-1)
                # print(label[:50])
                loss = self.nllloss(pred, label)

        if cfg.multi_acts_training and 'aspn_aug' in probs:
            prob = torch.cat(probs['aspn_aug'], 0)
            pred = prob.view(-1, prob.size(2))   #[B,T,Voov] -> [B*T, Voov]
            label = inputs['aspn_aug_4loss'].view(-1)
            # print(label.size())
            loss = self.nllloss(pred, label)
            total_loss += loss
            losses['aspn_aug'] = loss
        else:
            losses['aspn_aug'] = 0

        return total_loss, losses

    def spi_loss(self,inputs,probs,losses, label):
        bh_policies = inputs['bhProb']
        bh_policies = bh_policies
        mask = label>1e-7
        
        pas_probs = probs['pi(a|b)']
        pas_probs = pas_probs.permute(0,2,1)
        pas_probs = -self.nllloss_kd(pas_probs, label)
        pas_probs = torch.exp(pas_probs)
        
        
        pas_probs = mask * pas_probs
        pas_probs = torch.exp(torch.clamp(torch.sum(torch.log(torch.clamp(pas_probs,1e-10,1)),dim=-1,keepdims=True)/torch.sum(mask,dim=-1,keepdims=True),-80,80))
        bh_policies = torch.unsqueeze(bh_policies,dim=-1) 
        #torch.exp(torch.clamp(torch.sum(torch.log(torch.clamp(bh_policies,1e-10,1)),dim=-1,keepdims=True)/torch.sum(mask,dim=-1,keepdims=True),-80,80))
        
        sent_Qs = inputs['Q']
        sent_Qs = torch.unsqueeze(sent_Qs,dim=-1)
        
        IS = pas_probs/bh_policies
        
        stoch_loss = -1 * IS*sent_Qs
        stoch_loss = torch.mean(stoch_loss)
        delta = other_config['spi_penalty_coeff']
        
        #Not quite the constraint of TRPO
        stoch_const = torch.clamp(torch.mean(bh_policies*torch.log(bh_policies/torch.clamp(pas_probs,1e-7,1-1e-7)))-other_config['spi_penalty_coeff'],0,1e20)**2
        losses['stoch_loss'] = stoch_loss.detach().cpu().numpy()
        losses['stoch_const'] = stoch_const.detach().cpu().numpy()
        
        return other_config['spi_loss_wt']*stoch_loss + other_config['spi_const_wt']*stoch_const
    
    
    def forward(self, inputs, hidden_states, first_turn, mode, losses=None, probs=None):
        if mode == 'train' or mode == 'valid':
            # probs, hidden_states = \
            probs = \
                self.train_forward(inputs, hidden_states, first_turn)
            total_loss, losses = self.supervised_loss(inputs, probs)
            if cfg.enable_contrast:
                inputs_copy, hidden_states_copy = {}, {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs_copy[k] = v.clone().detach()
                    else:
                        inputs_copy[k] = v

                for k, v in hidden_states.items():
                    if isinstance(v, torch.Tensor):
                        hidden_states_copy[k] = v.clone().detach()
                    else:
                        hidden_states_copy[k] = v
                contrast_loss = self.contrast_loss(inputs_copy, hidden_states_copy, first_turn)
                total_loss += contrast_loss
                losses['contrast'] = contrast_loss
            return total_loss, losses
        elif mode == 'test':
            decoded = self.test_forward(inputs, hidden_states, first_turn)
            return decoded
        elif mode == 'rl_supervised':
            probs, decoded = \
                self.rl_forward(inputs, hidden_states, first_turn)
            total_loss, losses = self.supervised_loss(inputs, probs)
            #total_loss, losses = self.rl_loss(inputs, probs)
            return total_loss, losses, probs, decoded
        elif mode == 'rl_policy':
            pred = probs['aspn'].permute(0,2,1)
            label = inputs['aspn_4loss']
            seq_mask = (inputs['aspn_4loss']!=0)*1                        
            loss = self.nllloss_kd(pred, label)
            loss = torch.unsqueeze(inputs['G'],dim=1) * loss * 10 * cfg.rl_factor
            loss = torch.sum(loss)/torch.sum(seq_mask)
            losses['aspn'] = loss
            return loss, losses
            #raise NotImplementedError('RL not available at the moment')

    def contrast_loss(self, inputs, hidden_states, first_turn):
        
        """
        compute contrast loss for a single dialogue turn. 
        """
        def train_decode(model_name, init_hidden, hidden_states, probs, bidx=None, first_turn_override=None, return_prob_direct=False, input_name=None):
            # default model_name and input_name: bspn
            same_input = True if input_name == model_name else False
            if first_turn_override is None:
                first_turn_override=first_turn

            batch_size = inputs['user'].size(0) if bidx is None else len(bidx)
            dec_last_w = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[input_name])
            if bidx is None:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)
            else:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)[:, bidx, :]

            decode_step = inputs[input_name].size(1) if bidx is None else inputs['aspn_aug_batch'].size(1)
            hiddens = []
            sub_layer_hiddens = []
            for t in range(decode_step):
                # print('%s step %d'%(name, t))
                first_step = (t==0)
                if bidx is None:
                    dec_last_h = self.decoders[model_name](inputs, hidden_states, dec_last_w,
                                                                          dec_last_h, first_turn_override, first_step, return_bspn=same_input)
                    
                    if isinstance(dec_last_h, tuple):
                        dec_last_h,dect_sub_layer_h = dec_last_h
                        sub_layer_hiddens.append(dect_sub_layer_h)
                        
                    hiddens.append(dec_last_h)
                    dec_last_w = inputs[input_name][:, t].view(-1, 1)
                else:
                    assert input_name == 'aspn', 'only act span decoder support batch idx selection'
                    dec_last_h = self.decoders[model_name](inputs, hidden_states, dec_last_w,
                                                                                 dec_last_h, first_turn_override, first_step, bidx=bidx, return_bspn=same_input)
                    hiddens.append(dec_last_h)
                    dec_last_w = inputs['aspn_aug_batch'][:, t].view(-1, 1)

            dec_hs =  torch.cat(hiddens, dim=0).transpose(0,1)  # [1,B,H] ---> [B,T,H]
            if bidx is None:
                if return_prob_direct==True:
                    probs = self.decoders[model_name].get_probs(inputs, hidden_states, dec_hs, first_turn_override, return_bspn=same_input)
                else:
                    probs[input_name] = self.decoders[model_name].get_probs(inputs, hidden_states, dec_hs, first_turn_override, return_bspn=same_input)
                    if input_name != 'resp':
                        hidden_states[input_name] = dec_hs
            else:
                probs = self.decoders[model_name].get_probs(inputs, hidden_states, dec_hs, first_turn_override, bidx=bidx, return_bspn=same_input)
                
            if len(sub_layer_hiddens)>0:
                hidden_states[input_name+'_sub_layer'] = torch.cat(sub_layer_hiddens, dim=1)
            return hidden_states, probs
        
        def train_decode_multi(model_name, init_hidden, hidden_states, probs, bidx=None, first_turn_override=None, return_prob_direct=False, input_name=None, idx=None):
            # default model_name: bspn  input_name: cntfact_bspn
            # decode multi cntfact_bspn input for contrast loss
            same_input = True if input_name == model_name else False
            if first_turn_override is None:
                first_turn_override=first_turn

            batch_size = inputs['user'].size(0) if bidx is None else len(bidx)
            dec_last_w = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[input_name])
            if bidx is None:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)
            else:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)[:, bidx, :]

            decode_step = inputs[input_name][idx].size(1) if bidx is None else inputs['aspn_aug_batch'].size(1)
            hiddens = []
            sub_layer_hiddens = []
            for t in range(decode_step):
                # print('%s step %d'%(name, t))
                first_step = (t==0)
                if bidx is None:
                    dec_last_h = self.decoders[model_name](inputs, hidden_states, dec_last_w,
                                                                          dec_last_h, first_turn_override, first_step, return_bspn=same_input, idx=idx)
                    
                    if isinstance(dec_last_h, tuple):
                        dec_last_h,dect_sub_layer_h = dec_last_h
                        sub_layer_hiddens.append(dect_sub_layer_h)
                        
                    hiddens.append(dec_last_h)
                    dec_last_w = inputs[input_name][idx][:, t].view(-1, 1)
                else:
                    assert input_name == 'aspn', 'only act span decoder support batch idx selection'
                    dec_last_h = self.decoders[model_name](inputs, hidden_states, dec_last_w,
                                                                                 dec_last_h, first_turn_override, first_step, bidx=bidx, return_bspn=same_input, idx=idx)
                    hiddens.append(dec_last_h)
                    dec_last_w = inputs['aspn_aug_batch'][:, t].view(-1, 1)

            dec_hs =  torch.cat(hiddens, dim=0).transpose(0,1)  # [1,B,H] ---> [B,T,H]
            if input_name != 'resp':
                hidden_states[input_name + '_' + str(idx)] = dec_hs
            if len(sub_layer_hiddens)>0:
                hidden_states[input_name+'_sub_layer'] = torch.cat(sub_layer_hiddens, dim=1)
            return hidden_states, probs


        user_enc, user_enc_last_h = self.user_encoder(inputs['user']) # inputs: [B, T]
        usdx_enc, usdx_enc_last_h = self.usdx_encoder(inputs['usdx'])
        resp_enc, resp_enc_last_h = self.usdx_encoder(inputs['pv_resp'])
        user_enc = hidden_states['user']
        usdx_enc = hidden_states['usdx']
        resp_enc = hidden_states['resp']


        probs = {}

        if cfg.enable_contrast: # true
            bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.bspn_mode]) # compare with bspn
            hidden_states[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
            hidden_states, probs = train_decode(cfg.bspn_mode, init_hidden, hidden_states, probs, input_name=cfg.bspn_mode) 

            if cfg.enable_multi_cntfact:
                cntfact_bspn_enc = []
                for i in range(cfg.topk_cntfact):
                    _cntfact_bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.cntfact_bspn_mode][i])
                    cntfact_bspn_enc.append(_cntfact_bspn_enc)
                hidden_states[cfg.cntfact_bspn_mode] = cntfact_bspn_enc
                init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
                for i in range(cfg.topk_cntfact):
                    hidden_states, probs = train_decode_multi(cfg.bspn_mode, init_hidden, hidden_states, probs, input_name=cfg.cntfact_bspn_mode, idx=i)
                context = torch.mean(init_hidden, dim=0) # [2, B, H] -> [B, H]
                pos_hidden = torch.mean(hidden_states[cfg.bspn_mode], dim=1) # [B, T, H] -> [B, H]
                neg_hidden = [torch.mean(hidden_states[cfg.cntfact_bspn_mode + '_' + str(idx)], dim=1) for idx in range(cfg.topk_cntfact)]
                neg_hidden = torch.stack(neg_hidden, dim=1) # list of [B, H] -> [B, K, H]
                info_loss = InfoNCE(negative_mode='paired')
                loss = info_loss(context, pos_hidden, neg_hidden) * cfg.contrast_ratio
                
            else:
                cntfact_bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.cntfact_bspn_mode]) # compare with bspn
                hidden_states[cfg.cntfact_bspn_mode] = cntfact_bspn_enc
                init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
                hidden_states, probs = train_decode(cfg.bspn_mode, init_hidden, hidden_states, probs, input_name=cfg.cntfact_bspn_mode) 
                #hidden_states, probs = train_decode(cfg.cntfact_bspn_mode, init_hidden, hidden_states, probs, input_name=cfg.cntfact_bspn_mode) 
                context = torch.mean(init_hidden, dim=0) # [2, B, H] -> [B, H]
                pos_hidden = torch.mean(hidden_states[cfg.bspn_mode], dim=1) # [B, T, H] -> [B, H]
                neg_hidden = torch.mean(hidden_states[cfg.cntfact_bspn_mode], dim=1)
                info_loss = InfoNCE(negative_mode='unpaired')
                loss = info_loss(context, pos_hidden, neg_hidden) * cfg.contrast_ratio
            return loss 
        else:
            raise ValueError(f'Excepted enable_contrast all true, but got {cfg.enable_contrast}')
        

    def rl_loss(self, inputs, probs, decoded):
            def LabelSmoothingNLLLoss(logprob, labels):
                    return -(labels * logprob).sum((1,2)).mean()
            total_loss = 0
            if cfg.enable_cntfact: #TODO: eliminate bspn and bsdx here
                losses = {'bsdx':0, 'bspn':0, 'aspn':0, 'resp':0, 'dspn':0, 'cntfact_bsdx':0, 'cntfact_bspn':0}
            else:
                losses = {'bsdx':0, 'bspn':0, 'aspn':0, 'resp':0, 'dspn':0}
            for name, prob in probs.items():
                if name == 'aspn_aug':
                    continue
                # print(prob)
                # pred = torch.log(prob.view(-1, prob.size(2)))
                # print(pred[0, :50])
                if name != 'resp' or cfg.label_smoothing == .0:
                    if name == 'aspn' and ('L_det' in other_config['policy_loss'] or 'L_soft' in other_config['policy_loss']):
                        pred = prob.permute(0,2,1)
                        label = inputs[name+'_4loss']

                        seq_mask = (inputs['aspn_4loss']!=0)*1                        
                        loss = self.nllloss_kd(pred, label)
                        if 'L_det' in other_config['policy_loss']:
                            loss = torch.unsqueeze(inputs['G'],dim=1) * loss * 10
                            loss = torch.sum(loss)/torch.sum(seq_mask)
                        if 'L_sto' in other_config['policy_loss']:
                            loss += self.spi_loss(inputs, probs, losses, label)
                    else:
                        if name in ['pi(a|b)']:
                            continue
                                            
                        pred = prob.view(-1, prob.size(2))   #[B,T,Voov] -> [B*T, Voov]
                        label = inputs[name+'_4loss'].view(-1)
                        # print(label[:50])
                        loss = self.nllloss(pred, label)
                    total_loss += loss
                    losses[name] = loss
                else:
                    pred = prob.view(-1, prob.size(2))   #[B,T,Voov] -> [B*T, Voov]
                    label = inputs[name+'_4loss'].view(-1)
                    # print(label[:50])
                    loss = self.nllloss(pred, label)

            if cfg.multi_acts_training and 'aspn_aug' in probs:
                prob = torch.cat(probs['aspn_aug'], 0)
                pred = prob.view(-1, prob.size(2))   #[B,T,Voov] -> [B*T, Voov]
                label = inputs['aspn_aug_4loss'].view(-1)
                # print(label.size())
                loss = self.nllloss(pred, label)
                total_loss += loss
                losses['aspn_aug'] = loss
            else:
                losses['aspn_aug'] = 0

            return total_loss, losses

    def train_forward(self, inputs, hidden_states, first_turn):
        
        def train_pi_a_b(inputs, hidden_states, probs, usdx_enc_last_h):
            probs['pi(a|b)'] = []
            hidden_states_bkups = {'usdx':False, 
                                  cfg.bspn_mode:False, 
                                  'user':False, 
                                  'aspn':False}
            inputs_bkups = {'usdx':True, 
                           'user':True, 
                           'pv_resp':True, 
                           'pv_'+cfg.bspn_mode:True,
                           'db':False,
                           'user_onehot':False,
                           'user_nounk':True,
                           'pv_aspn':True
                           }
            probs_bkups = {
                cfg.bspn_mode: False,
                'aspn':False
                }
            if cfg.enable_dspn:
                hidden_states_bkups['dspn']=False
                inputs_bkups['pv_dspn']=True
                probs_bkups['dspn']=False
                
            
            self.back_up_values(hidden_states, hidden_states_bkups)
            self.back_up_values(inputs, inputs_bkups)
            self.back_up_values(probs, probs_bkups)
            
            if cfg.enable_dspn:
                dspn_enc, _ = self.span_encoder(cuda_(torch.ones(inputs['pv_dspn'].shape, dtype=torch.long)))
                hidden_states['dspn'] = dspn_enc
                hidden_states, probs = self.train_decode('dspn', usdx_enc_last_h, hidden_states, probs)
            
            if cfg.enable_bspn:
                bspn_enc, _ = self.span_encoder(cuda_(torch.ones(inputs['pv_'+cfg.bspn_mode].shape, dtype=torch.long)))
                hidden_states[cfg.bspn_mode] = bspn_enc
                
                user_enc_dummy, user_enc_last_h_dummy = self.user_encoder(cuda_(torch.zeros(inputs['user'].shape, dtype=torch.long)))
                usdx_enc_dummy, usdx_enc_last_h_dummy = self.usdx_encoder(cuda_(torch.zeros(inputs['usdx'].shape, dtype=torch.long)))
        
                init_hidden_dummy = user_enc_last_h_dummy if cfg.bspn_mode == 'bspn' else usdx_enc_last_h_dummy
                hidden_states, probs = train_decode(cfg.bspn_mode, init_hidden_dummy, hidden_states, probs,first_turn_override=True, bidx=None)
                
            aspn_enc, _ = self.span_encoder(inputs['pv_aspn'])
            hidden_states['aspn'] = aspn_enc
            
            _, prob_aspn = train_decode('aspn', usdx_enc_last_h_dummy, hidden_states, probs,first_turn_override=True,return_prob_direct=True)
            probs['pi(a|b)'] = prob_aspn
                
            self.restore_back_up_values(hidden_states, hidden_states_bkups)
            self.restore_back_up_values(inputs, inputs_bkups)
            self.restore_back_up_values(probs, probs_bkups)
            
            
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        """
        def train_decode(name, init_hidden, hidden_states, probs, bidx=None, first_turn_override=None, return_prob_direct=False):
            if first_turn_override is None:
                first_turn_override=first_turn

            batch_size = inputs['user'].size(0) if bidx is None else len(bidx)
            dec_last_w = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[name])
            if bidx is None:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)
            else:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)[:, bidx, :]

            decode_step = inputs[name].size(1) if bidx is None else inputs['aspn_aug_batch'].size(1)
            hiddens = []
            sub_layer_hiddens = []
            for t in range(decode_step):
                # print('%s step %d'%(name, t))
                first_step = (t==0)
                if bidx is None:
                    if cfg.enable_contrast and name == 'bspn':
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                            dec_last_h, first_turn_override, first_step, return_bspn=True)
                    else:
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                            dec_last_h, first_turn_override, first_step)
                    
                    if isinstance(dec_last_h, tuple):
                        dec_last_h,dect_sub_layer_h = dec_last_h
                        sub_layer_hiddens.append(dect_sub_layer_h)
                        
                    hiddens.append(dec_last_h)
                    dec_last_w = inputs[name][:, t].view(-1, 1)
                else:
                    assert name == 'aspn', 'only act span decoder support batch idx selection'
                    if cfg.enable_contrast and name == 'bspn':
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                                dec_last_h, first_turn_override, first_step, return_bspn=True)
                    
                    else:
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                                dec_last_h, first_turn_override, first_step, bidx=bidx)
                    hiddens.append(dec_last_h)
                    dec_last_w = inputs['aspn_aug_batch'][:, t].view(-1, 1)

            dec_hs =  torch.cat(hiddens, dim=0).transpose(0,1)  # [1,B,H] ---> [B,T,H]
            if bidx is None:
                if return_prob_direct==True:
                    probs = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override)
                else:
                    if cfg.enable_contrast and name == 'bspn':
                        probs[name] = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override, return_bspn=True)
                    else:
                        probs[name] = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override)
                    if name != 'resp':
                        hidden_states[name] = dec_hs
            else:
                probs = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override, bidx=bidx)
                
            if len(sub_layer_hiddens)>0:
                hidden_states[name+'_sub_layer'] = torch.cat(sub_layer_hiddens, dim=1)
            return hidden_states, probs


        user_enc, user_enc_last_h = self.user_encoder(inputs['user'])
        usdx_enc, usdx_enc_last_h = self.usdx_encoder(inputs['usdx'])
        resp_enc, resp_enc_last_h = self.usdx_encoder(inputs['pv_resp'])
        hidden_states['user'] = user_enc
        hidden_states['usdx'] = usdx_enc
        hidden_states['resp'] = resp_enc

        probs = {}
        #pdb.set_trace()
        if cfg.use_bcq:
            inputs['bcq_state'] = copy.deepcopy(inputs[cfg.bspn_mode])
            inputs['bcq_action'] = self.bcq.generate_action(inputs['bcq_state'])

        if cfg.enable_dspn: # false
            dspn_enc, _ = self.span_encoder(inputs['pv_dspn'])
            hidden_states['dspn'] = dspn_enc
            hidden_states, probs = train_decode('dspn', usdx_enc_last_h, hidden_states, probs)

        if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward: # true
            bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.bspn_mode]) # pv_bsdx
            hidden_states[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
            hidden_states, probs = train_decode(cfg.bspn_mode, init_hidden, hidden_states, probs) # hidden_states: dict ['user', 'usdx', 'resp', 'bsdx']; prob: dict ['bsdx']
        
        elif cfg.enable_cntfact and not cfg.enable_contrast: # true
            cntfact_bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.cntfact_bspn_mode]) # compare with bspn
            hidden_states[cfg.cntfact_bspn_mode] = cntfact_bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
            hidden_states, probs = train_decode(cfg.cntfact_bspn_mode, init_hidden, hidden_states, probs) 
        elif cfg.enable_contrast:
            bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.bspn_mode]) # compare with bspn
            hidden_states[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
            hidden_states, probs = train_decode(cfg.bspn_mode, init_hidden, hidden_states, probs) 
            #del probs[cfg.cntfact_bspn_mode] # don't count cntfact bspn in loss calculation


        if cfg.enable_aspn: # true
            aspn_enc, _ = self.span_encoder(inputs['pv_aspn'])
            hidden_states['aspn'] = aspn_enc
            hidden_states, probs = train_decode('aspn', usdx_enc_last_h, hidden_states, probs) # hidden_states, prob: add ['aspn']

        hidden_states, probs = train_decode('resp', usdx_enc_last_h, hidden_states, probs) # update ['resp']
        
        if 'L_sto' in other_config['policy_loss']:
            train_pi_a_b(inputs, hidden_states, probs, usdx_enc_last_h)
            
        if cfg.enable_dst and cfg.bspn_mode == 'bsdx': # False
            bspn_enc, _ = self.span_encoder(inputs['pv_bspn'])
            hidden_states['bspn'] = bspn_enc
            hidden_states, probs = train_decode('bspn', user_enc_last_h, hidden_states, probs)

        if cfg.enable_aspn and cfg.multi_acts_training and 'aspn_aug' in inputs:
            probs['aspn_aug'] = []
            batch_size = inputs['user'].size(0)

            for b in range(len(inputs['aspn_bidx'])//batch_size+1):
                bidx_batch = inputs['aspn_bidx'][b*batch_size : (b+1)*batch_size]
                if bidx_batch:
                    inputs['aspn_aug_batch'] = inputs['aspn_aug'][b*batch_size : (b+1)*batch_size, :]
                    _, ps = train_decode('aspn', usdx_enc_last_h, hidden_states, None, bidx=bidx_batch)
                    probs['aspn_aug'].append(ps)

        return probs # dict ['bsdx', 'aspn', 'resp']
    
    def rl_forward(self, inputs, hidden_states, first_turn):
        
        def train_pi_a_b(inputs, hidden_states, probs, usdx_enc_last_h):
            probs['pi(a|b)'] = []
            hidden_states_bkups = {'usdx':False, 
                                  cfg.bspn_mode:False, 
                                  'user':False, 
                                  'aspn':False}
            inputs_bkups = {'usdx':True, 
                           'user':True, 
                           'pv_resp':True, 
                           'pv_'+cfg.bspn_mode:True,
                           'db':False,
                           'user_onehot':False,
                           'user_nounk':True,
                           'pv_aspn':True
                           }
            probs_bkups = {
                cfg.bspn_mode: False,
                'aspn':False
                }
            if cfg.enable_dspn:
                hidden_states_bkups['dspn']=False
                inputs_bkups['pv_dspn']=True
                probs_bkups['dspn']=False
                
            
            self.back_up_values(hidden_states, hidden_states_bkups)
            self.back_up_values(inputs, inputs_bkups)
            self.back_up_values(probs, probs_bkups)
            
            if cfg.enable_dspn:
                dspn_enc, _ = self.span_encoder(cuda_(torch.ones(inputs['pv_dspn'].shape, dtype=torch.long)))
                hidden_states['dspn'] = dspn_enc
                hidden_states, probs = self.train_decode('dspn', usdx_enc_last_h, hidden_states, probs)
            
            if cfg.enable_bspn:
                bspn_enc, _ = self.span_encoder(cuda_(torch.ones(inputs['pv_'+cfg.bspn_mode].shape, dtype=torch.long)))
                hidden_states[cfg.bspn_mode] = bspn_enc
                
                user_enc_dummy, user_enc_last_h_dummy = self.user_encoder(cuda_(torch.zeros(inputs['user'].shape, dtype=torch.long)))
                usdx_enc_dummy, usdx_enc_last_h_dummy = self.usdx_encoder(cuda_(torch.zeros(inputs['usdx'].shape, dtype=torch.long)))
        
                init_hidden_dummy = user_enc_last_h_dummy if cfg.bspn_mode == 'bspn' else usdx_enc_last_h_dummy
                hidden_states, probs = train_decode(cfg.bspn_mode, init_hidden_dummy, hidden_states, probs,first_turn_override=True, bidx=None, return_decoded=False)
                
            aspn_enc, _ = self.span_encoder(inputs['pv_aspn'])
            hidden_states['aspn'] = aspn_enc
            
            _, prob_aspn = train_decode('aspn', usdx_enc_last_h_dummy, hidden_states, probs,first_turn_override=True,return_prob_direct=True, return_decoded=False)
            probs['pi(a|b)'] = prob_aspn
                
            self.restore_back_up_values(hidden_states, hidden_states_bkups)
            self.restore_back_up_values(inputs, inputs_bkups)
            self.restore_back_up_values(probs, probs_bkups)
            
            
        """
        compute required outputs for a single dialogue turn. Turn state{Dict} will be updated in each call.
        """
        def train_decode(name, init_hidden, hidden_states, probs, decoded=None, bidx=None, first_turn_override=None, return_prob_direct=False, return_decoded=True, decoded_mode='greedy'):
            '''
            param:
            decoded_mode: corresponding to test mode's greedy method, only implement greedy now.
            '''
            if first_turn_override is None:
                first_turn_override=first_turn

            batch_size = inputs['user'].size(0) if bidx is None else len(bidx)
            dec_last_w = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[name])
            if bidx is None:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)
            else:
                dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)[:, bidx, :]

            decode_step = inputs[name].size(1) if bidx is None else inputs['aspn_aug_batch'].size(1)
            if return_decoded:
                decode_idx = []
            hiddens = []
            sub_layer_hiddens = []
            for t in range(decode_step):
                # print('%s step %d'%(name, t))
                first_step = (t==0)
                if bidx is None:
                    if cfg.enable_contrast and name == 'bspn':
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                            dec_last_h, first_turn_override, first_step, return_bspn=True)
                    else:
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                            dec_last_h, first_turn_override, first_step)
                    # calculate turn_prob for turn top_k calculation
                    dec_hs = dec_last_h.transpose(0,1)
                    if cfg.enable_contrast and name == 'bspn':
                        prob_turn = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn, return_bspn=True)  #[B,1,V_oov]
                    else:
                        prob_turn = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn)  #[B,1,V_oov]
                    
                    if isinstance(dec_last_h, tuple):
                        dec_last_h,dect_sub_layer_h = dec_last_h
                        sub_layer_hiddens.append(dect_sub_layer_h)
                        
                    hiddens.append(dec_last_h)
                    if not self.teacher_forcing_decode[name]:
                        if not self.limited_vocab_decode[name]:
                            dec_last_w_infer = torch.topk(prob_turn.squeeze(1), 1)[1]
                        else:
                            for b in range(batch_size):
                                w = int(dec_last_w[b].cpu().numpy())
                                if name == 'aspn':
                                    mask = self.reader.aspn_masks_tensor[w]
                                elif name == 'bspn' or name == 'bsdx' or name == 'cntfact_bspn' or name == 'cntfact_bsdx':
                                    mask = self.reader.bspn_masks_tensor[w]
                                prob_turn[b][0][mask] += 100
                            dec_last_w_infer = torch.topk(prob_turn.squeeze(1), 1)[1]
                    else:
                        if t < inputs[name].size(1):
                            dec_last_w_infer = inputs[name][:, t].view(-1, 1)
                        else:
                            dec_last_w_infer = cuda_(torch.zeros(batch_size, 1).long())
                    dec_last_w = inputs[name][:, t].view(-1, 1)
                else:
                    assert name == 'aspn', 'only act span decoder support batch idx selection'
                    if cfg.enable_contrast and name == 'bspn':
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                                dec_last_h, first_turn_override, first_step, return_bspn=True)
                    
                    else:
                        dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                                dec_last_h, first_turn_override, first_step, bidx=bidx)
                    hiddens.append(dec_last_h)
                    dec_last_w = inputs['aspn_aug_batch'][:, t].view(-1, 1)
                if return_decoded:
                    decode_idx.append(dec_last_w_infer.view(-1).clone())
                    #dec_last_w[dec_last_w>=self.vocab_size] = 2

            dec_hs =  torch.cat(hiddens, dim=0).transpose(0,1)  # [1,B,H] ---> [B,T,H]
            if bidx is None:
                if return_prob_direct==True:
                    probs = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override)
                else:
                    if cfg.enable_contrast and name == 'bspn':
                        probs[name] = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override, return_bspn=True)
                    else:
                        probs[name] = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override)
                    if name != 'resp':
                        hidden_states[name] = dec_hs
            else:
                probs = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn_override, bidx=bidx)
                
            if len(sub_layer_hiddens)>0:
                hidden_states[name+'_sub_layer'] = torch.cat(sub_layer_hiddens, dim=1)
            
            if return_decoded:
                decoded_np= torch.stack(decode_idx, dim=1).cpu().numpy()
                for sidx, seq in enumerate(decoded_np):
                    try:
                        eos = list(seq).index(self.eos_idx[name])
                        decoded_np[sidx, eos+1:] = 0
                    except:
                        continue
                decoded[name] = [list(_) for _ in decoded_np] 
                return hidden_states, probs, decoded
            else:
                return hidden_states, probs


        user_enc, user_enc_last_h = self.user_encoder(inputs['user'])
        usdx_enc, usdx_enc_last_h = self.usdx_encoder(inputs['usdx'])
        resp_enc, resp_enc_last_h = self.usdx_encoder(inputs['pv_resp'])
        hidden_states['user'] = user_enc
        hidden_states['usdx'] = usdx_enc
        hidden_states['resp'] = resp_enc

        probs, decoded = {}, {}
        #pdb.set_trace()

        if cfg.use_bcq:
            inputs['bcq_state'] = copy.deepcopy(inputs[cfg.bspn_mode])
            inputs['bcq_action'] = self.bcq.generate_action(inputs['bcq_state'])
        if cfg.enable_dspn: # false
            dspn_enc, _ = self.span_encoder(inputs['pv_dspn'])
            hidden_states['dspn'] = dspn_enc
            hidden_states, probs, decoded = train_decode('dspn', usdx_enc_last_h, hidden_states, probs, decoded)

        if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward: # true
            bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.bspn_mode]) # pv_bsdx
            hidden_states[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
            hidden_states, probs, decoded = train_decode(cfg.bspn_mode, init_hidden, hidden_states, probs, decoded) # hidden_states: dict ['user', 'usdx', 'resp', 'bsdx']; prob: dict ['bsdx']
        
        elif cfg.enable_cntfact and not cfg.enable_contrast: # true
            cntfact_bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.cntfact_bspn_mode]) # compare with bspn
            hidden_states[cfg.cntfact_bspn_mode] = cntfact_bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
            hidden_states, probs, decoded = train_decode(cfg.cntfact_bspn_mode, init_hidden, hidden_states, probs, decoded) 
        elif cfg.enable_contrast:
            bspn_enc, _ = self.span_encoder(inputs['pv_'+cfg.bspn_mode]) # compare with bspn
            hidden_states[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h #cfg.bspn_mode='bsdx'
            hidden_states, probs, decoded = train_decode(cfg.bspn_mode, init_hidden, hidden_states, probs, decoded) 
            #del probs[cfg.cntfact_bspn_mode] # don't count cntfact bspn in loss calculation


        if cfg.enable_aspn: # true
            aspn_enc, _ = self.span_encoder(inputs['pv_aspn'])
            hidden_states['aspn'] = aspn_enc
            hidden_states, probs, decoded = train_decode('aspn', usdx_enc_last_h, hidden_states, probs, decoded) # hidden_states, prob: add ['aspn']

        hidden_states, probs, decoded = train_decode('resp', usdx_enc_last_h, hidden_states, probs, decoded) # update ['resp']
        
        train_pi_a_b(inputs, hidden_states, probs, usdx_enc_last_h)
            
        if cfg.enable_dst and cfg.bspn_mode == 'bsdx': # False
            bspn_enc, _ = self.span_encoder(inputs['pv_bspn'])
            hidden_states['bspn'] = bspn_enc
            hidden_states, probs, decoded = train_decode('bspn', user_enc_last_h, hidden_states, probs, decoded)

        if cfg.enable_aspn and cfg.multi_acts_training and 'aspn_aug' in inputs:
            'Not implemented now'
            probs['aspn_aug'] = []
            batch_size = inputs['user'].size(0)

            for b in range(len(inputs['aspn_bidx'])//batch_size+1):
                bidx_batch = inputs['aspn_bidx'][b*batch_size : (b+1)*batch_size]
                if bidx_batch:
                    inputs['aspn_aug_batch'] = inputs['aspn_aug'][b*batch_size : (b+1)*batch_size, :]
                    _, ps = train_decode('aspn', usdx_enc_last_h, hidden_states, None, bidx=bidx_batch)
                    probs['aspn_aug'].append(ps)

        return probs, decoded # dict ['bsdx', 'aspn', 'resp']

    def back_up_values(self, dict_, name_is_idx_dict):
        for name,is_idx in name_is_idx_dict.items():
            self.back_up_value(dict_, name, is_idx=is_idx)
    
    def back_up_value(self, dict_, name, is_idx=None):
        dict_[name+'_bkup'] = dict_[name]
        if is_idx:
            dict_[name]=cuda_(torch.zeros(dict_[name].shape, dtype=torch.long))
        else:
            dict_[name]=cuda_(torch.zeros(dict_[name].shape))
        
    def restore_back_up_values(self, dict_, name_is_idx_dict):
        for name in name_is_idx_dict.keys():
            self.restore_back_up_value(dict_, name)
        
    def restore_back_up_value(self, dict_, name):
        dict_[name] = dict_[name+'_bkup']
        dict_.pop(name+'_bkup',None)
        
    def test_forward(self, inputs, hs, first_turn):
        user_enc, user_enc_last_h = self.user_encoder(inputs['user'])
        usdx_enc, usdx_enc_last_h = self.usdx_encoder(inputs['usdx'])
        resp_enc, resp_enc_last_h = self.usdx_encoder(inputs['pv_resp'])
        hs['user'] = user_enc
        hs['usdx'] = usdx_enc
        hs['resp'] = resp_enc

        decoded = {}
        if cfg.use_bcq:
            inputs['bcq_state'] = copy.deepcopy(inputs[cfg.bspn_mode])
            inputs['bcq_action'] = self.bcq.generate_action(inputs['bcq_state'])

        if cfg.enable_dst and cfg.bspn_mode == 'bsdx':
            bspn_enc, _ = self.span_encoder(inputs['pv_bspn'])
            hs['bspn'] = bspn_enc
            hs, decoded = self.greedy_decode('bspn', user_enc_last_h, first_turn, inputs, hs, decoded)

        if cfg.enable_dspn:
            dspn_enc, dspn_enc_last_h = self.span_encoder(inputs['pv_dspn'])
            hs['dspn'] = dspn_enc
            hs, decoded = self.greedy_decode('dspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)

        if cfg.enable_bspn and not cfg.enable_cntfact or cfg.enable_cntfact_reward:
            bspn_enc, bspn_enc_last_h = self.span_encoder(inputs['pv_'+cfg.bspn_mode])
            hs[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h
            hs, decoded = self.greedy_decode(cfg.bspn_mode, init_hidden, first_turn, inputs, hs, decoded)

            if not cfg.use_true_db_pointer and 'bspn' in decoded:
                for bi, bspn_list in enumerate(decoded['bspn']):
                    turn_domain = inputs['turn_domain'][bi]
                    db_ptr = self.reader.bspan_to_DBpointer(bspn_list, turn_domain)
                    book_ptr = 'cannot be predicted, use the groud truth'
                    inputs['db_np'][bi, :cfg.pointer_dim-2] = db_ptr
                inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())

        elif cfg.enable_cntfact and not cfg.enable_contrast:
            cntfact_bspn_enc, cntfact_bspn_enc_last_h = self.span_encoder(inputs['pv_'+cfg.cntfact_bspn_mode])
            hs[cfg.cntfact_bspn_mode] = cntfact_bspn_enc
            init_hidden = user_enc_last_h if cfg.cntfact_bspn_mode == 'cntfact_bspn' else usdx_enc_last_h
            hs, decoded = self.greedy_decode(cfg.cntfact_bspn_mode, init_hidden, first_turn, inputs, hs, decoded)

            if not cfg.use_true_db_pointer and 'cntfact_bspn' in decoded:
                for bi, cntfact_bspn_list in enumerate(decoded['cntfact_bspn']):
                    turn_domain = inputs['turn_domain'][bi]
                    db_ptr = self.reader.bspan_to_DBpointer(cntfact_bspn_list, turn_domain)
                    book_ptr = 'cannot be predicted, use the groud truth'
                    inputs['db_np'][bi, :cfg.pointer_dim-2] = db_ptr
                inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())

        elif cfg.enable_contrast:
            bspn_enc, bspn_enc_last_h = self.span_encoder(inputs['pv_'+cfg.bspn_mode])
            hs[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h
            hs, decoded = self.greedy_decode(cfg.bspn_mode, init_hidden, first_turn, inputs, hs, decoded)
            if not cfg.use_true_db_pointer and 'bspn' in decoded:
                for bi, bspn_list in enumerate(decoded['bspn']):
                    turn_domain = inputs['turn_domain'][bi]
                    db_ptr = self.reader.bspan_to_DBpointer(bspn_list, turn_domain)
                    book_ptr = 'cannot be predicted, use the groud truth'
                    inputs['db_np'][bi, :cfg.pointer_dim-2] = db_ptr
                inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())


        if cfg.enable_aspn:
            aspn_enc, aspn_enc_last_h = self.span_encoder(inputs['pv_aspn'])
            hs['aspn'] = aspn_enc
            if cfg.aspn_decode_mode == 'greedy':
                hs, decoded = self.greedy_decode('aspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)
            elif cfg.aspn_decode_mode == 'beam':
                if cfg.record_mode:
                    hs_nbest, decoded_nbest = self.beam_decode('aspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)
                else:
                    hs, decoded = self.beam_decode('aspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)
            elif 'sampling' in cfg.aspn_decode_mode:
                hs, decoded = self.sampling_decode('aspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)

        if cfg.record_mode:
            batch_size = inputs['user'].size(0)
            self.reader.aspn_collect, self.reader.resp_collect = [], []
            for i in range(batch_size):
                self.reader.aspn_collect.append([])
                self.reader.resp_collect.append([])
            for i in range(cfg.nbest):
                self.reader.resp_collect.append([])
                wid_seqs_np = decoded_nbest.cpu().numpy()
                inputs['aspn_np'] = wid_seqs_np[:, i, :]
                update_input('aspn', inputs)
                hs['aspn'] = hs_nbest[:, i, :, :]
                hs, decoded = self.greedy_decode('resp', usdx_enc_last_h, first_turn, inputs, hs, decoded)
                for b in range(batch_size):
                    self.reader.resp_collect[b].append(decoded['resp'][b])
                    self.reader.aspn_collect[b].append(list(inputs['aspn_np'][b][:]))
        else:
            hs, decoded = self.greedy_decode('resp', usdx_enc_last_h, first_turn, inputs, hs, decoded)

        return decoded

    def addActSelection(self):
        for p in self.parameters():
            p.requires_grad=False
        self.act_selection = ActSelectionModel(cfg.hidden_size, cfg.max_span_length, cfg.nbest)


    def RL_forward(self, inputs, decoded, hiddens_batch, decoded_batch):
        """[summary]
        :param hiddens_batch: [B, nbest, T, H]
        :param decoded_batch: [B, nbest, T]
        """
        batch_size = hiddens_batch.size()[0]
        logprob = self.act_selection(hiddens_batch)   #[B, nbest]
        dis = Categorical(torch.exp(logprob))
        action = dis.sample()
        index = action.view(-1).cpu().numpy().to_list()
        loss = 0
        for b in range(batch_size):
            ref = self.reader.vocab.sentence_decode(inputs['aspn_np'][b], eos='<eos_a>')
            ref_acts= self.reader.aspan_to_act_list(ref)
            select = self.reader.vocab.sentence_decode(decoded['aspn'][index][b], eos='<eos_a>')
            select_acts= self.reader.aspan_to_act_list(select)
            reward = utils.f1_score(ref_acts, select_acts)
            loss += reward * logprob[b, index]
        return loss

    def greedy_decode(self, name, init_hidden, first_turn, inputs, hidden_states, decoded, return_bspn=False):
        max_len = cfg.max_nl_length if name == 'resp' else cfg.max_span_length
        batch_size = inputs['user'].size(0)
        dec_last_w = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[name])
        dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)
        hiddens, decode_idx = [], []
        for t in range(max_len):
            # print('%s step %d'%(name, t))
            first_step = (t==0)
            if cfg.enable_contrast and name=='bspn':
                dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                         dec_last_h, first_turn, first_step, mode='test', return_bspn=True)

            else:    
                dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                         dec_last_h, first_turn, first_step, mode='test')
            dec_hs = dec_last_h.transpose(0,1)
            if cfg.enable_contrast and name == 'bspn':
                prob_turn = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn, return_bspn=True)  #[B,1,V_oov]
            else:
                prob_turn = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn)  #[B,1,V_oov]
            hiddens.append(dec_last_h)

            if not self.teacher_forcing_decode[name]:
                if not self.limited_vocab_decode[name]:
                    dec_last_w = torch.topk(prob_turn.squeeze(1), 1)[1]
                else:
                    for b in range(batch_size):
                        w = int(dec_last_w[b].cpu().numpy())
                        if name == 'aspn':
                            mask = self.reader.aspn_masks_tensor[w]
                        elif name == 'bspn' or name == 'bsdx' or name == 'cntfact_bspn' or name == 'cntfact_bsdx':
                            mask = self.reader.bspn_masks_tensor[w]
                        prob_turn[b][0][mask] += 100
                    dec_last_w = torch.topk(prob_turn.squeeze(1), 1)[1]
            else:
                if t < inputs[name].size(1):
                    dec_last_w = inputs[name][:, t].view(-1, 1)
                else:
                    dec_last_w = cuda_(torch.zeros(batch_size, 1).long())

            decode_idx.append(dec_last_w.view(-1).clone())
            dec_last_w[dec_last_w>=self.vocab_size] = 2

        hidden_states[name] =  torch.cat(hiddens, dim=0).transpose(0,1)  # [1,B,H] ---> [B,T,H]
        decoded_np= torch.stack(decode_idx, dim=1).cpu().numpy()
        for sidx, seq in enumerate(decoded_np):
            try:
                eos = list(seq).index(self.eos_idx[name])
                decoded_np[sidx, eos+1:] = 0
            except:
                continue
        decoded[name] = [list(_) for _ in decoded_np]   #[B,T]

        if name != 'resp':
            inputs[name+'_np'] = decoded_np
            update_input(name, inputs)

        return hidden_states, decoded

    def beam_decode(self, name, init_hidden, first_turn, inputs, hidden_states, decoded):
        beam_width = self.beam_width
        nbest = self.nbest  # how many sentence do you want to generate
        decoded_batch, hiddens_batch = [], []

        batch_size = inputs['user'].size(0)

        dec_last_w_batch = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[name])
        dec_last_h_batch = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)   #[1,B,H]
        hiddens, decode_idx = [], []

        for bidx in range(batch_size):
            dec_last_w = dec_last_w_batch[bidx, :].unsqueeze(1)   #[1,1]
            dec_last_h = dec_last_h_batch[:, bidx, :].unsqueeze(1)   #[1,1,H]

            # Number of sentence to generate
            endnodes = []
            number_required = min((nbest + 1), nbest - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(dec_last_h, None, dec_last_w, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(cfg.beam_diverse_param), node))
            qsize = 1

            first_step = True
            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                dec_last_w = n.wordid
                dec_last_h = n.h
                # print(dec_last_w.size())
                # print(dec_last_h.size())

                if n.wordid.item() == self.eos_idx[name] and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                    dec_last_h, first_turn, first_step, bidx=[bidx], mode='test')
                dec_h = dec_last_h.transpose(0,1)
                prob_turn = self.decoders[name].get_probs(inputs, hidden_states, dec_h, first_turn, bidx=[bidx])  #[B,1,V_oov]

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_probs, dec_last_ws = torch.topk(prob_turn.squeeze(1), beam_width)

                for new_k in range(beam_width):
                    rank = new_k
                    decoded_t = dec_last_ws[0][new_k].view(1, -1).clone()
                    log_p = log_probs[0][new_k].item()

                    node = BeamSearchNode(dec_last_h, n, decoded_t, n.logp + log_p, n.leng + 1, rank)
                    score = -node.eval(cfg.beam_diverse_param)
                    try:
                        nodes.put((score, node))
                    except:
                        # very rarely but truely exists cases that different sequences have a same score
                        # which lead to a can't-comparing exception
                        continue

                # increase qsize
                qsize += beam_width - 1
                first_step = False

            # choose nbest paths, back trace them
            if len(endnodes) < nbest:
                endnodes += [nodes.get() for _ in range(nbest - len(endnodes))]

            wid_seqs = []
            hiddens = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                decode_idx, hs = [], []
                decode_idx.append(n.wordid)
                hs.append(n.h)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    decode_idx.append(n.wordid)
                    hs.append(n.h)
                zeros = cuda_(torch.zeros(1, cfg.max_span_length - len(decode_idx)).long())
                decoded_T = torch.cat(decode_idx[::-1] + [zeros], dim=1)   # [1,1] ---> [1,T]
                zeros = cuda_(torch.zeros(1, cfg.max_span_length - len(decode_idx), hs[0].size(2)))
                hs = torch.cat(hs[::-1] + [zeros], dim=1)   # [1,1,H] ---> [1,T,H]
                wid_seqs.append(decoded_T)
                hiddens.append(hs) # [nbest,1,H]

            wid_seqs = torch.stack(wid_seqs, dim=0)   #[nbest, 1,T]
            hiddens = torch.stack(hiddens, dim=0)   #[nbest, 1,T, H]
            decoded_batch.append(wid_seqs)
            hiddens_batch.append(hiddens)

        hiddens_batch = torch.cat(hiddens_batch, dim=1).transpose(0,1)   #[B, nbest, T, H]
        decoded_batch = torch.cat(decoded_batch, dim=1).transpose(0,1)   #[B, nbest, T]
        if cfg.record_mode == False:
            hidden_states[name], inputs[name+'_np'] = self.aspn_selection(inputs, decoded, hiddens_batch,
                                                                                                                  decoded_batch)
            update_input(name, inputs)
            decoded[name] = [list(_) for _ in inputs[name+'_np']]
            return hidden_states, decoded
        else:
            decoded[name] = [list(_) for _ in decoded_batch.cpu().numpy()[:, cfg.nbest-1, :]]
            return hiddens_batch, decoded_batch

        # if cfg.use_true_pv_resp:
        #     resp_enc, resp_enc_last_h = self.usdx_encoder(inputs['resp'])
        #     hidden_states['resp'] = resp_enc

    def sampling_decode(self, name, init_hidden, first_turn, inputs, hidden_states, decoded):
        max_len = cfg.max_nl_length if name == 'resp' else cfg.max_span_length
        batch_size = inputs['user'].size(0)

        decoded_batch = []
        hiddens_batch = []
        for s in range(cfg.nbest):
            # print('nbest:', s)
            dec_last_w = cuda_(torch.ones(batch_size, 1).long() * self.go_idx[name])
            dec_last_h = (init_hidden[-1]+init_hidden[-2]).unsqueeze(0)
            hiddens, decode_idx = [], []
            for t in range(max_len):
                # print('%s step %d'%(name, t))
                first_step = (t==0)
                dec_last_h = self.decoders[name](inputs, hidden_states, dec_last_w,
                                                                             dec_last_h, first_turn, first_step, mode='test')
                dec_hs = dec_last_h.transpose(0,1)
                prob_turn = self.decoders[name].get_probs(inputs, hidden_states, dec_hs, first_turn)  #[B,1,V_oov]
                hiddens.append(dec_last_h)   #list of [1, B, H] of length T

                if cfg.aspn_decode_mode == 'topk_sampling':
                    logprobs, topk_words = torch.topk(prob_turn.squeeze(1), cfg.topk_num)
                    widx = torch.multinomial(torch.exp(logprobs), 1, replacement=True)
                    dec_curr_w = torch.gather(topk_words, 1, widx)
                    for b in range(batch_size):
                        if dec_last_w[b].item() == 8 or dec_last_w[b].item() == 0:
                            dec_curr_w[b] = 0
                    dec_last_w = dec_curr_w.clone()
                elif cfg.aspn_decode_mode == 'nucleur_sampling':
                    logprobs, topk_words = torch.topk(prob_turn.squeeze(1), 55)   #55 is enough for valid aspn tokens
                    probs = torch.exp(logprobs)
                    dec_curr_w = []
                    for b in range(batch_size):
                        for pnum in range(1, 55):
                            if torch.sum(probs[b][:pnum]) >= cfg.nucleur_p:
                                break
                        sample = torch.multinomial(probs[b][:pnum], 1, replacement=True)
                        if dec_last_w[b].item() == 8 or dec_last_w[b].item() == 0:
                            dec_curr_w.append(cuda_(torch.zeros(1).long()))
                        else:
                            dec_curr_w.append(topk_words[b][sample])
                    dec_last_w = torch.stack(dec_curr_w, 0)

                decode_idx.append(dec_last_w.view(-1).clone())   #list of [B] of length T
                dec_last_w[dec_last_w>=self.vocab_size] = 2

            decoded_np= torch.stack(decode_idx, dim=1)   #[B, T]
            hiddens_batch.append(torch.cat(hiddens, dim=0).transpose(0,1))   #list of [B, T, H] of length nbest
            decoded_batch.append(decoded_np)

        hiddens_batch = torch.stack(hiddens_batch, dim=1)   #[B, nbest, T, H]
        decoded_batch = torch.stack(decoded_batch, dim=1)   #[B, nbest, T]
        hidden_states[name], inputs[name+'_np'] = self.aspn_selection(inputs, decoded, hiddens_batch,
                                                                                                              decoded_batch)

        update_input(name, inputs)
        decoded[name] = [list(_) for _ in inputs[name+'_np']]
        # print(decoded[name][0][0:5])
        # print(decoded[name][1][0:5])
        # print(decoded[name][2][0:5])

        return hidden_states, decoded

    def aspn_selection(self, inputs, decoded, hiddens_batch, decoded_batch):
        """[summary]
        :param hiddens_batch: [B, nbest, T, H]
        :param decoded_batch: [B, nbest, T]
        """
        batch_size = inputs['user'].size(0)
        wid_seqs_np = decoded_batch.cpu().numpy()  #[B, nbest, T]
        decoded['aspn'] = []
        multi_acts = []
        for i in range(cfg.nbest):
            decoded['aspn'].append([list(_) for _ in wid_seqs_np[:, i, :]])
        if cfg.act_selection_scheme == 'high_test_act_f1':
            decode_chosen = []
            hidden_chosen = []
            for b in range(batch_size):
                ref = self.reader.vocab.sentence_decode(inputs['aspn_np'][b], eos='<eos_a>')
                ref_acts= self.reader.aspan_to_act_list(ref)
                scores = []
                acts = ''
                for i in range(self.nbest):
                    decode_str = self.reader.vocab.sentence_decode(decoded['aspn'][i][b], eos='<eos_a>')
                    decode_str_acts= self.reader.aspan_to_act_list(decode_str)
                    acts += decode_str + ' | '
                    f1 = utils.f1_score(ref_acts, decode_str_acts)
                    # print(decode_str, f1)
                    scores.append(f1)
                multi_acts.append(acts[:-3])
                max_score_idx = scores.index(max(scores))
                decode_chosen.append(decoded_batch[b][max_score_idx])
                hidden_chosen.append(hiddens_batch[b][max_score_idx])

            hidden_chosen = torch.stack(hidden_chosen, dim=0)   #[B, T, H]
            decode_chosen = torch.stack(decode_chosen, dim=0).cpu().numpy()   #[B,T]
            self.reader.multi_acts_record = multi_acts   #[B, T]
        else:
            hidden_chosen = hiddens_batch[:, 0, :, :]   #[B, nbest, T, H]
            decode_chosen = wid_seqs_np[:, 0, :]
        return hidden_chosen, decode_chosen


    def RL_train(self, inputs, hs, hiddens_batch, decoded_batch, first_turn):
        """[summary]
        :param hiddens_batch: [B, nbest, T, H]
        :param decoded_batch: [B, nbest, T]
        """
        user_enc, user_enc_last_h = self.user_encoder(inputs['user'])
        usdx_enc, usdx_enc_last_h = self.usdx_encoder(inputs['usdx'])
        resp_enc, resp_enc_last_h = self.usdx_encoder(inputs['pv_resp'])
        hs['user'] = user_enc
        hs['usdx'] = usdx_enc
        hs['resp'] = resp_enc

        decoded = {}

        if cfg.enable_dst and cfg.bspn_mode == 'bsdx':
            bspn_enc, _ = self.span_encoder(inputs['pv_bspn'])
            hs['bspn'] = bspn_enc
            hs, decoded = self.greedy_decode('bspn', user_enc_last_h, first_turn, inputs, hs, decoded)

        if cfg.enable_dspn:
            dspn_enc, dspn_enc_last_h = self.span_encoder(inputs['pv_dspn'])
            hs['dspn'] = dspn_enc
            hs, decoded = self.greedy_decode('dspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)

        if cfg.enable_bspn:
            bspn_enc, bspn_enc_last_h = self.span_encoder(inputs['pv_'+cfg.bspn_mode])
            hs[cfg.bspn_mode] = bspn_enc
            init_hidden = user_enc_last_h if cfg.bspn_mode == 'bspn' else usdx_enc_last_h
            hs, decoded = self.greedy_decode(cfg.bspn_mode, init_hidden, first_turn, inputs, hs, decoded)

            if not cfg.use_true_db_pointer and 'bspn' in decoded:
                for bi, bspn_list in enumerate(decoded['bspn']):
                    turn_domain = inputs['turn_domain'][bi]
                    db_ptr = self.reader.bspan_to_DBpointer(bspn_list, turn_domain)
                    book_ptr = 'cannot be predicted, use the groud truth'
                    inputs['db_np'][bi, :cfg.pointer_dim-2] = db_ptr
                inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())

            aspn_enc, aspn_enc_last_h = self.span_encoder(inputs['pv_aspn'])
            hs['aspn'] = aspn_enc
            if cfg.aspn_decode_mode == 'greedy':
                hs, decoded = self.greedy_decode('aspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)
            elif cfg.aspn_decode_mode == 'beam':
                hs, decoded = self.beam_decode('aspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)
            elif 'sampling' in cfg.aspn_decode_mode:
                hs, decoded = self.sampling_decode('aspn', usdx_enc_last_h, first_turn, inputs, hs, decoded)



def update_input(name, inputs):
    inputs[name+'_unk_np'] = copy.deepcopy(inputs[name+'_np'])
    inputs[name+'_unk_np'][inputs[name+'_unk_np']>=cfg.vocab_size] = 2   # <unk>
    inputs[name+'_onehot'] = get_one_hot_input(inputs[name+'_unk_np'])
    inputs[name] = cuda_(torch.from_numpy(inputs[name+'_unk_np']).long())
    inputs[name+'_nounk'] = cuda_(torch.from_numpy(inputs[name+'_np']).long())


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length, rank=None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length
        self.rank = rank if rank is not None else 0

    def __lt__(self, other):
        return self.rank < other.rank

    def eval(self, alpha=0):
        reward = self.rank
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) - alpha * reward

    def print_node(self):
        string = str(self.wordid_oov.item())
        node = self.prevNode
        while node != None:
            string = str(nn.wordid_oov.item()) + ',' + string
            node = node.prevNode
        print(string)

class SharedEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, cfg.embed_size)
        self.span_encoder = biGRUencoder(self.embedding)
        #self.l1 = nn.Linear(cfg.hidden_size, cfg.hidden_size) 
        #self.l1 = nn.Linear(state_dim + action_dim, 400)
        #self.l2 = nn.Linear(400, 300)

    def forward(self, x):
        return self.span_encoder(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super().__init__()
        #self.encoder = shared_encoder
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.phi = phi


    def forward(self, state, action):
        #encoded = self.encoder(state, action)
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        #a = self.phi * self.max_action * torch.tanh(self.l3(a))
        # no clip
        a = self.phi * torch.tanh(self.l3(a))
        #return (a + action).clamp(-self.max_action, self.max_action)
        return (a + action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        #self.encoder = shared_encoder
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, state, action):
        #encoded = self.encoder(state, action)
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super().__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            #z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
            # no clip
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
        


class BCQ(nn.Module):
    def __init__(self, reader, state_dim=cfg.hidden_size, action_dim=cfg.hidden_size, max_action=1, device='cpu', discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        super().__init__()
        latent_dim = action_dim * 2
        self.reader = reader
        self.vocab = self.reader.vocab
        self.vocab_size = self.vocab.vocab_size
        self.vsize_oov = self.vocab.vocab_size_oov
        self.embed_size = cfg.embed_size
        self.hidden_size = cfg.hidden_size

        self.shared_encoder = SharedEncoder(self.vocab_size).to(device)
        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device


    def select_action(self, state):		
        with torch.no_grad():
            state_embed, state_last_h = self.shared_encoder(state)
            #state_last_h = self.scale_to_range(torch.sum(state_last_h, dim=0))
            state_last_h = torch.sum(state_last_h, dim=0)

            state_repeated = state_last_h.repeat_interleave(100, dim=0)
            actions = self.actor(state_repeated, self.vae.decode(state_repeated))
            
            q_values = self.critic.q1(state_repeated, actions)
            q_values = q_values.view(-1, 100)
            
            max_indices = q_values.argmax(dim=1)
            actions = actions.view(-1, 100, actions.size(-1))
            selected_actions = actions[torch.arange(actions.size(0)), max_indices]
            return selected_actions
        
    def scale_to_range(self, tensor, range_min=-1, range_max=1):
        """
        scale tensor into range 。
        
        para:
        - tensor: [batch_size, hidden_size]
        - range_min
        - range_max
        
        return:
        - tensor_scaled
        """
        # 按批次（dim=0）找到最小值和最大值
        min_vals = torch.min(tensor, dim=1, keepdim=True)[0]
        max_vals = torch.max(tensor, dim=1, keepdim=True)[0]
        
        tensor_std = (tensor - min_vals) / (max_vals - min_vals)
        tensor_scaled = tensor_std * (range_max - range_min) + range_min
        
        return tensor_scaled
           


    def train_forward(self, state, action, next_state, reward, not_done, batch_size=100):
        #TODO: add normalization
        #for it in range(iterations):
        # Sample replay buffer / batch
        #state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Variational Auto-Encoder Training
        state_embed, state_last_h = self.shared_encoder(state)
        action_embed, action_last_h = self.shared_encoder(action)
        next_state_embed, next_state_last_h = self.shared_encoder(next_state)
        # no clip
        #state_last_h = self.scale_to_range(torch.sum(state_last_h, dim=0))
        state_last_h = (torch.sum(state_last_h, dim=0))
        #action_last_h = self.scale_to_range(torch.sum(action_last_h, dim=0))
        action_last_h = (torch.sum(action_last_h, dim=0))
        #next_state_last_h = self.scale_to_range(torch.sum(next_state_last_h, dim=0))
        next_state_last_h = (torch.sum(next_state_last_h, dim=0))
        recon, mean, std = self.vae(state_last_h, action_last_h)
        recon_loss = F.mse_loss(recon, action_last_h)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optimizer.zero_grad()
        #vae_loss.backward()
        vae_loss.backward(retain_graph=True)
        self.vae_optimizer.step()


        # Critic Training
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state_last_h = torch.repeat_interleave(next_state_last_h, 10, 0)

            # Compute value of perturbed actions sampled from the VAE
            target_Q1, target_Q2 = self.critic_target(next_state_last_h, self.actor_target(next_state_last_h, self.vae.decode(next_state_last_h)))

            # Soft Clipped Double Q-learning 
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

            target_Q = reward + not_done * self.discount * target_Q

        #action_last_h_critic = action_last_h.clone().detach().requires_grad_(True)
        current_Q1, current_Q2 = self.critic(state_last_h, action_last_h)
        #current_Q1, current_Q2 = self.critic(state_last_h, action_last_h_critic)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        #critic_loss.backward()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()


        # Pertubation Model / Action Training
        sampled_actions = self.vae.decode(state_last_h)
        perturbed_actions = self.actor(state_last_h, sampled_actions)

        # Update through DPG
        actor_loss = -self.critic.q1(state_last_h, perturbed_actions).mean()
            
        self.actor_optimizer.zero_grad()
        #actor_loss.backward()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()


        # Update Target Networks 
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return vae_loss, actor_loss, critic_loss
    
    def valid_forward(self, state, action, next_state, reward, not_done, batch_size=100):
        with torch.no_grad():  # 确保在这个上下文中不计算梯度
            # Variational Auto-Encoder Validation
            state_embed, state_last_h = self.shared_encoder(state)
            action_embed, action_last_h = self.shared_encoder(action)
            next_state_embed, next_state_last_h = self.shared_encoder(next_state)
            #state_last_h = self.scale_to_range(torch.sum(state_last_h, dim=0))
            #action_last_h = self.scale_to_range(torch.sum(action_last_h, dim=0))
            #next_state_last_h = self.scale_to_range(torch.sum(next_state_last_h, dim=0))
            state_last_h = (torch.sum(state_last_h, dim=0))
            action_last_h = (torch.sum(action_last_h, dim=0))
            next_state_last_h = (torch.sum(next_state_last_h, dim=0))
            recon, mean, std = self.vae(state_last_h, action_last_h)
            recon_loss = F.mse_loss(recon, action_last_h)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            # Critic Validation
            # Duplicate next state 10 times
            next_state_last_h = torch.repeat_interleave(next_state_last_h, 10, 0)

            # Compute value of perturbed actions sampled from the VAE
            target_Q1, target_Q2 = self.critic_target(next_state_last_h, self.actor_target(next_state_last_h, self.vae.decode(next_state_last_h)))

            # Soft Clipped Double Q-learning 
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

            target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state_last_h, action_last_h)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Pertubation Model / Action Validation
            sampled_actions = self.vae.decode(state_last_h)
            perturbed_actions = self.actor(state_last_h, sampled_actions)

            # Update through DPG
            actor_loss = -self.critic.q1(state_last_h, perturbed_actions).mean()
        
        # 返回计算的损失，用于评估
        return vae_loss.item(), actor_loss.item(), critic_loss.item()


