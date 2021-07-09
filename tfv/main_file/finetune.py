import os
import torch
import argparse
import random 
import numpy as np
import time
from operator import itemgetter
import pandas as pd


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
LL = 256


class ModelDefine(nn.Module):
    def __init__(self,use_structure):
        super(ModelDefine, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_dp)
        self.use_vm = use_structure
        dim = 768
        self.classification_liner = nn.Linear(dim, 2)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, w_idxs1, type_idxs, mask_idxs=None, vm_low=None, vm_upper=None):
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
        last_layer = hidden_states

        max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2), kernel_size=last_layer.size(1)).squeeze(-1)
        max_pooling_fts = self.drop_out(max_pooling_fts)
        return self.classification_liner(max_pooling_fts)

class ModelDefine_tapas(nn.Module):
    def __init__(self,use_structure):
        super(ModelDefine_tapas, self).__init__()
        self.tapas = TapasModel.from_pretrained('google/tapas-base')
        self.use_vm = use_structure
        dim = 768
        self.classification_liner = nn.Linear(dim, 2)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, w_idxs1, token_type_ids, mask_idxs=None, vm_low=None, vm_upper=None):
        
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
        last_layer = hidden_states

        max_pooling_fts = F.max_pool1d(last_layer.transpose(1, 2), kernel_size=last_layer.size(1)).squeeze(-1)
        max_pooling_fts = self.drop_out(max_pooling_fts)
        return self.classification_liner(max_pooling_fts)

class Model:
    def __init__(self,dt,device,use_structure,use_tapas):
        lr=2e-5
        self.dt = dt
        self.device = device
        self.use_tapas = use_tapas
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
        self.use_vm = use_structure

    def train(self,output_finetune_path):
        for i in range(20):
            print("===" * 10)
            print("epoch%d" % i)
            for batch_id, batch in tqdm(enumerate(self.dt.iter_batches(which="train")), ncols=80):
                batch_size = len(batch[0])
                self.model.train()
                input_ids, type_idxs, vms_low, vms_upper, labels = [
                    Variable(e).long().to(self.device) for e in batch]
                logits = self.model(input_ids, type_idxs, input_ids > 0.5, vms_low, vms_upper) 
                loss = F.cross_entropy(logits, labels)
                self.train_loss.update(loss.item(), batch_size)
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 20)
                self.optimizer.step()
                self.updates += 1

            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset()
            self.validate(epoch=i, output_finetune_path=output_finetune_path, which='dev')
            self.validate(epoch=i, output_finetune_path=output_finetune_path, which='test')
            self.validate(epoch=i, output_finetune_path=output_finetune_path, which='simple')
            self.validate(epoch=i, output_finetune_path=output_finetune_path, which='complex')

    def validate(self, epoch, output_finetune_path, which="dev"):
        #predicted_labels_lns = []
        def simple_accuracy(preds1, labels1):
            correct_num = sum([1.0 if p1 == p2 else 0.0 for p1, p2 in zip(preds1, labels1)])
            return correct_num / len(preds1)

        preds_label = []
        gold_label = []
        y_predprob = []
        for batch in tqdm(self.dt.iter_batches(which=which), ncols=80):
            batch_size = len(batch[0])
            self.model.eval()
            input_ids, type_idxs, vms_low, vms_upper, labels = [
                Variable(e).long().to(self.device) for e in batch]
            logits = self.model(input_ids, type_idxs, input_ids > 0.5, vms_low, vms_upper) 
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            preds_label.extend(preds)
            gold_label.extend(batch[-1])
            y_predprob.extend(F.softmax(logits, dim=1).detach().cpu().numpy()[:, -1])
        '''
        for p_label, g_label, s in zip(preds_label, gold_label, y_predprob):
            predicted_labels_lns.append(u'{}\t{}\t{}\t{}\n'.format(p_label, g_label, s, 'error' if g_label != p_label else 'correct'))
        open('prediction_result_of_{}.txt'.format(which), "w").writelines(predicted_labels_lns)
        '''
        preds_label = np.array(preds_label)
        gold_label = np.array(gold_label)
        acc = simple_accuracy(preds_label, gold_label)
        p, r, f1, _ = precision_recall_fscore_support(gold_label, preds_label)
        print("{} acc={}, p={}, r={}, f1={}".format(which, acc, p, r, f1))
        
        '''
        if which == "dev" and epoch == 19:
            if self.use_vm == "no":
                if not os.path.exists(output_finetune_path+"/Ftab"):
                    os.mkdir(output_finetune_path+"/Ftab")
                self.save(output_finetune_path+"/Ftab/epoch_{}_acc_{}.pt".format(epoch, acc), epoch)  # 跑一下模型。
                self.model_pt = output_finetune_path+"/Ftab/epoch_{}_acc_{}.pt".format(epoch, acc)
            else:
                if not os.path.exists(output_finetune_path+"/Fsat"):
                    os.mkdir(output_finetune_path+"/Fsat")
                self.save(output_finetune_path+"/Fsat/epoch_{}_acc_{}.pt".format(epoch, acc), epoch)  # 跑一下模型。
                self.model_pt = output_finetune_path+"/Fsat/epoch_{}_acc_{}.pt".format(epoch, acc)
        '''



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

    def test(self, inp=None):
        input_ids, type_idxs, vms_low, vms_upper = inp
        logits = self.model(input_ids.to(self.device), type_idxs.to(self.device), (input_ids > 0.5).to(self.device), vms_low.to(self.device), vms_upper.to(self.device)) 
        m = nn.Softmax(dim=1)
        output = m(logit)
        score = output.detach().cpu().numpy()[0][1]
        pred = np.argmax(logit.detach().cpu().numpy(), axis=1)[0]
        return pred,score 


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

def build_vmm(source_ids, low_or_upper):
    vm = []
    for rowid, colid in source_ids:
        if rowid == -1 and colid == -1:
            vm.append([1] * LL)
            continue
        row_see = []
        for rowid2, colid2 in source_ids:
            if rowid2 == -1 and colid2 == -1:
                row_see.append(1)
            elif rowid2 == -2 and colid2 == -2:  # padding
                row_see.append(0)
            else:
                if rowid2 == 0:
                    row_see.append(1)
                else:
                    if (low_or_upper == 'low' and rowid == rowid2) or (
                            low_or_upper == 'upper' and (rowid == rowid2 or colid == colid2)):
                        row_see.append(1)
                    else:
                        row_see.append(0)
        assert len(row_see) == len(source_ids), "incorrect number"
        vm.append(row_see)
    return vm

def get_inputs(dt,claim,tab_id):
    table = dt.tables[table_id]
    q_tokenized = dt.tokenizer_bert.tokenize(claim)
    q_tokenized = ["[CLS]"] + q_tokenized + ['[SEP]']
    token_ids = dt.tokenizer_bert.convert_tokens_to_ids(q_tokenized)
    token_ids_source = [(-1, -1)] * len(q_tokenized)
    for rowid, content in enumerate(table_content):
        info = [dt.tokenizer_bert.tokenize(i) for i in content]
        for cell_id, cell in enumerate(content):
            token_ids.extend(dt.tokenizer_bert.convert_tokens_to_ids(cell))
            token_ids_source.extend([(rowid, cell_id)] * len(cell))
    if len(token_ids) > LL:
        token_ids = token_ids[:LL]
        token_ids_source = token_ids_source[:LL]
    else:
        token_ids += [pad_token] * (LL - len(token_ids))
        token_ids_source += [(-2, -2)] * (LL - len(token_ids_source))

    if self.pretrain_mode == "tapas":
        table_contents_raw = dt.tables_raw[tab_id]
        table_df = pd.DataFrame(table_contents_raw[1:], columns=table_contents_raw[0]).astype(str)
        inputs_tapas = self.tokenizer_tapas(table=table_df, queries=claim, padding="max_length", truncation=True, return_tensors="pt")
        type_ids = inputs_tapas['token_type_ids']
    else:
        type1_len = len(q_tokenized)
        type_ids = [0] * type1_len + [1] * (LL - type1_len)
    return torch.tensor(token_ids).view(1,LL), torch.tensor(type_ids).view(1,LL), torch.tensor(build_vmm(token_ids_source, 'low')).view(1,LL,LL), torch.tensor(
            build_vmm(token_ids_source, 'upper')).view(1,LL,LL)


def finetune_run(dt,device,use_structure,intermediate_model,output_finetune_path,use_tapas):
    model = Model(dt,device,use_structure,use_tapas)
    if intermediate_model!="":
        model.resume(intermediate_model)
    model.train(output_finetune_path)
    return model.model_pt


def score_table_entailment(dt,device,use_structure,finetune_path,use_tapas):
    model = Model(dt,device,use_structure,use_tapas)
    model.resume(finetune_path)
    model.validate(epoch=-1,  output_finetune_path=finetune_path, which='simple')
    model.validate(epoch=-1,  output_finetune_path=finetune_path, which='complex')
    model.validate(epoch=-1,  output_finetune_path=finetune_path, which='dev')
    model.validate(epoch=-1,  output_finetune_path=finetune_path, which='test')


def predict_table_entailment(dt,device,use_structure,claim,tab_id,finetune_path,use_tapas):
    model = Model(dt,device,use_structure,use_tapas)
    model.resume(finetune_path)
    inp = get_inputs(dt,claim,tab_id)
    pred,score  = model.test(inp)
    if pred==1:
        print("prediction: True")
    else:
        print("prediction: False")
    print("score_True: "+str(score))

    
def score_verification(dt,device,use_structure,finetune_path,k,use_tapas):
    model = Model(dt,device,use_structure,use_tapas)
    model.resume(finetune_path)
    all_num = 0
    hit_num = 0
    for data in dt.verification_dataset:
        all_num = all_num+1
        claim = data["claim"]
        tab_id = data["tab_id"]
        q_tokenized = dt.tokenizer.tokenize(claim)
        remain_tables,_ = filter_tables(dt,q_tokenized)
        inputs = []
        for tab_id in remain_tables:
            inputs.append(get_inputs(dt,claim,tab_id))
        res = []
        for inp in inputs:
            tab_id = inp[-1]
            pred,score  = model.test(inp)
            res.append({"tab_id":tab_id,"pred":pred,"score":score})
        res = sorted(res,key = itemgetter('score') ,reverse = True)
        flag = 0
        max_id = k
        if len(res)<max_id:
            max_id = len(res)
        for i in range(0,max_id):
            if res[i]["pred"]==1 and res[i]["tab_id"]==tab_id:
                flag = 1
                break
        if flag==1:
            hit_num = hit_num+1
    print("hit_P: "+str(hit_num)+"/"+str(all_num))


def filter_tables(dt,q_tokenized):
    len_all=0
    remain_tables = []
    for tab_id in dt.tables:
        len_all = len_all+1
        table_content = dt.tables[tab_id]
        flag = 0
        _, _, caption = dt.all_dat[tab_id]
       
        if len(list(set(caption)&set(q_tokenized))) < 1:
            continue
        
        for colid in range(max([len(content) for content in table_content])):
            for rowid, content in enumerate(table_content):
                cell = content[colid]
                if list(set(cell)&set(q_tokenized)):
                    flag = flag + 1
                    break
            if flag>5:
                break
        if flag>5:
            remain_tables.append(tab_id)
    return remain_tables,len_all

def predict_verification(dt,device,use_structure,claim,finetune_path,csv_path,k,use_tapas):
    t1=time.time()
    model = Model(dt,device,use_structure,use_tapas)
    model.resume(finetune_path)
    remain_tables = []
    len_all = 0
    len_remain = 0
    q_tokenized = dt.tokenizer.tokenize(claim)

    print("---------- filtering tables ------------")
    remain_tables,len_all = filter_tables(dt,q_tokenized)
  
    print("table_all: " + str(len_all))
    print("table_remain: " + str(len(remain_tables)))

    print("---------- creating features ------------")
    inputs = []
    for tab_id in remain_tables:
        inputs.append(get_inputs(dt,claim,tab_id))

    print("---------- runing model ------------")
    res = []
    for inp in inputs:
        tab_id = inp[-1]
        pred,score  = model.test(inp)
        res.append({"tab_id":tab_id,"pred":pred,"score":score})

    print("---------- sorting  ------------")
    res = sorted(res,key = itemgetter('score') ,reverse = True)

    print("---------- printing top-"+str(k)+" results  ------------")
    for i in range(0,k):
        if res[i]["pred"]==1:
            print("tab_id: "+ res[i]["tab_id"] +"; pred: True"+"; score_True: "+str(res[i]["score"]))
        else:
            print("tab_id: "+ res[i]["tab_id"] +"; pred: False"+"; score_True: "+str(res[i]["score"]))

    print("---------- printing top-1 CSV  ------------")
    x = pd.read_csv(csv_path+'/'+res[0]["tab_id"],sep="#")
    print(x)
    t2=time.time()
    print("total_time: "+str(t2-t1))