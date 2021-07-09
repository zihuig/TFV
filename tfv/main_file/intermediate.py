import os
import torch
import argparse
import random 
import numpy as np
import time 

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np

import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from torch.nn import CrossEntropyLoss, MSELoss

default_collate_func = dataloader.default_collate
LL = 256

def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

import torch.optim as optim

from transformers import BertModel, BertTokenizer, AutoConfig, TapasModel, TapasForSequenceClassification, TapasTokenizer, TapasConfig

from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F



bert_model_dp = 'bert-base-uncased'
pad_token = 0


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

ACT2FN = {
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



class ModelDefine(nn.Module):
    def __init__(self,use_structure):
        super(ModelDefine, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_dp)
        self.use_vm = use_structure
        self.config = AutoConfig.from_pretrained(bert_model_dp)
        self.cls = BertOnlyMLMHead(self.config)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(self, w_idxs1, type_idxs, mask_idxs=None, vm_low=None, vm_upper=None, labels=None):

        embedding_output = self.bert.embeddings(w_idxs1, type_idxs * (0 if pad_token else 1))

        extended_attention_mask_base = mask_idxs.long().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = vm_low.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_low = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask = vm_upper.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_upper = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask = extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_baseline = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = embedding_output
        for ly_idx, layer_module in enumerate(self.bert.encoder.layer):
            if ly_idx < 6:
                extended_attention_mask = extended_attention_mask_low
            else:
                extended_attention_mask = extended_attention_mask_upper
            if self.use_vm == 'no':
                extended_attention_mask = extended_attention_mask_baseline
            hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]
        sequence_output = hidden_states
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),  labels.view(-1))

        return masked_lm_loss

class ModelDefine_tapas(nn.Module):
    def __init__(self,use_structure):
        super(ModelDefine_tapas, self).__init__()
        self.tapas = TapasModel.from_pretrained('google/tapas-base')
        self.use_vm = use_structure
        dim = 768
        self.drop_out = nn.Dropout(0.1)
        self.config = self.tapas.config
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, w_idxs1, token_type_ids, mask_idxs=None, vm_low=None, vm_upper=None, labels=None):

        
        embedding_output = self.tapas.embeddings(w_idxs1, token_type_ids.reshape(token_type_ids.shape[0],token_type_ids.shape[2],token_type_ids.shape[3]))

        extended_attention_mask_base = mask_idxs.long().unsqueeze(1).unsqueeze(2)
        extended_attention_mask = vm_low.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_low = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask = vm_upper.unsqueeze(1) * extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_upper = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask = extended_attention_mask_base
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_baseline = (1.0 - extended_attention_mask) * -10000.0

        hidden_states = embedding_output
        for ly_idx, layer_module in enumerate(self.tapas.encoder.layer):
            if ly_idx < 6:
                extended_attention_mask = extended_attention_mask_low
            else:
                extended_attention_mask = extended_attention_mask_upper
            if self.use_vm == 'no':
                extended_attention_mask = extended_attention_mask_baseline
            hidden_states = layer_module(hidden_states, attention_mask=extended_attention_mask)[0]
        sequence_output = hidden_states
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),  labels.view(-1))

        return masked_lm_loss


class Model:
    def __init__(self,dt,device,use_structure,use_tapas):
        lr=2e-5
        self.dt = dt
        self.device = device
        self.use_tapas = use_tapas
        self.use_vm = use_structure
        if use_tapas == "yes":
            self.model =ModelDefine_tapas(use_structure)
        else:
            self.model = ModelDefine(use_structure)
        #         self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.train_loss = AverageMeter()
        self.updates = 0
        self.optimizer = optim.Adamax([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.model_pt = None


    def train(self,output_intermediate_path):
        model_pt = ""
        for i in range(20):
            print("epoch%d" % i)
            for batch_id, batch in tqdm(enumerate(self.dt.iter_batches_mlm()), ncols=80):
                start_time = time.time()
                batch_size = len(batch[0])
                self.model.train()
                input_ids, type_idxs, vms_low, vms_upper, labels = [
                    Variable(e).long().to(self.device) for e in batch]
                loss = self.model(input_ids, type_idxs, input_ids > 0.5, vms_low, vms_upper,labels)
                self.train_loss.update(loss.item(), batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.updates += 1
            self.train_loss.reset()
            
            if self.use_vm == "no":
                if not os.path.exists(output_intermediate_path+"/tab_mlm_cells"):
                    os.mkdir(output_intermediate_path+"/tab_mlm_cells")
                self.save(output_intermediate_path+"/tab_mlm_cells/epoch_{}.pt".format(i),i)  # 跑一下模型。
                model_pt = output_intermediate_path+"/tab_mlm_cells/epoch_{}.pt".format(i)
            else:
                if not os.path.exists(output_intermediate_path+"/sat_mlm_cells"):
                    os.mkdir(output_intermediate_path+"/sat_mlm_cells")
                self.save(output_intermediate_path+"/sat_mlm_cells/epoch_{}.pt".format(i),i)  # 跑一下模型。
                model_pt = output_intermediate_path+"/sat_mlm_cells/epoch_{}.pt".format(i)
        self.model_pt = model_pt



    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'epoch': epoch
        }
        torch.save(params, filename)

    def resume(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        state_dict = checkpoint['state_dict']

        new_state = set(self.model.state_dict().keys())
        for k in list(state_dict['network'].keys()):
            if k not in new_state:
                del state_dict['network'][k]
        self.model.load_state_dict(state_dict['network'], strict=False)
        self.model.to(self.device)
        return self.model


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mlm_run(dt,device,use_structure,output_intermediate_path,use_tapas):
    model = Model(dt,device,use_structure,use_tapas)
    model.train(output_intermediate_path)
    return model.model_pt



