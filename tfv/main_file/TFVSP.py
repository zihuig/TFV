
from intermediate import mlm_run
from finetune import finetune_run,score_table_entailment,predict_table_entailment,score_verification,predict_verification
import os
from transformers import BertModel, BertTokenizer, TapasTokenizer, TapasModel
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import torch
import pandas as pd
import pickle
import numpy as np
import argparse

LL = 256

bert_model_dp = 'bert-base-uncased'
tapas_model_dp = 'google/tapas-base'
pad_token = 0

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_mode',
                    type=str,
                    choices=['bert','tapas'],
                    default='bert')

parser.add_argument('--intermediate_mode',
                    type=str,
                    choices=['no','mlm','sat-mlm'],
                    default='sat-mlm')

parser.add_argument('--finetune_mode',
                    type=str,
                    choices=['classification','sat-classification'],
                    default='sat-classification')

parser.add_argument('--output_path',
                    type=str,
                    default='../../train_model')

parser.add_argument('--csv_path',
                    type=str,
                    default='../../data/all_csv/')

parser.add_argument('--claim_path',
                    type=str,
                    default='../../collected_data')

parser.add_argument('--dataset_path',
                    type=str,
                    default='../../data')

parser.add_argument('--cuda_id',
                    type=str,
                    default='0')

parser.add_argument('--k',
                    type=int,
                    default=5)


class DataManager:
    def __init__(self, csv_path, claim_path, dataset_path, pretrain_mode, intermediate_mode):
        
        self.tokenizer_bert = BertTokenizer.from_pretrained(bert_model_dp)
        self.pretrain_mode = pretrain_mode
        self.intermediate_mode = intermediate_mode
        if self.pretrain_mode=="tapas":
            self.tokenizer_tapas = TapasTokenizer.from_pretrained(tapas_model_dp)
            self.tokenizer_tapas.model_max_length = LL
            
        tab_f = f"tables_and_count_info.cache"
        tab_f_raw = f"tables_raw.cache"
        self.mlm_probability = 0.15
        self.verification_dataset = []

        if os.path.exists(tab_f):
            print('load tables from cache', tab_f)
            self.tables = pickle.load(open(tab_f, 'rb'))
            print('load tables_raw from cache', tab_f)
            self.tables_raw = pickle.load(open(tab_f_raw, 'rb'))
        else:
            print("creat ",tab_f)
            print("creat ",tab_f_raw)
            self.tables,self.tables_raw = self.load_the_tables(csv_path)
            pickle.dump(self.tables, open(tab_f, 'wb'))
            pickle.dump(self.tables_raw, open(tab_f_raw, 'wb'))
        
        self.all_dat = self.load_all_the_dat(claim_path)

        train_smps_ = self.get_dat(dataset_path,'train')
        test_smps_ = self.get_dat(dataset_path,'test')
        valid_smps_ = self.get_dat(dataset_path,'val')
        train_smps = self.trans2ids(train_smps_)
        test_smps = self.trans2ids(test_smps_)
        valid_smps = self.trans2ids(valid_smps_)

        self.train_dataset = MyDataSet(train_smps,pretrain_mode)
        self.dev_dataset = MyDataSet(valid_smps,pretrain_mode)
        self.test_dataset = MyDataSet(test_smps,pretrain_mode)
        self.simple_smps = MyDataSet(self.trans2ids(self.get_dat(dataset_path,'simple')),pretrain_mode)
        self.complex_smps = MyDataSet(self.trans2ids(self.get_dat(dataset_path,'complex')),pretrain_mode)

        if intermediate_mode!="no":
            mlm_smps_  = self.get_mlm_dat(dataset_path)
            mlm_smps_ = self.trans2ids_mlm(mlm_smps_)
            self.mlm_dataset = MyDataSet_mlm(mlm_smps_,pretrain_mode)
        

    def load_the_tables(self,dp):
        table_contents = {}
        table_raw_contents = {}
        for fnm in tqdm(os.listdir(dp)):
            table_contents[fnm],table_raw_contents[fnm] = self.load_a_table(dp + fnm)
        return table_contents,table_raw_contents

    def load_a_table(self, fnm):
        def add_num_info(raw_table_contents, table_contents):
            from collections import defaultdict
            col_digit_flag = defaultdict(set)
            for content in raw_table_contents[1:]:
                for colid, col in enumerate(content):
                    col_digit_flag[colid].add(col.isdigit())
            digit_cols = []
            for colid, flags in col_digit_flag.items():
                if len(flags) == 1 and True in flags:
                    digit_cols.append(colid)

            for colid in digit_cols:
                col_infos = [content[colid] for content in raw_table_contents]
                idxs = sorted(range(1, len(col_infos)), key=lambda idx: float(col_infos[idx]) * -1)
                for rank_num, idx in enumerate(idxs):
                    table_contents[idx][colid].append("[unused{}]".format(rank_num))
            return table_contents

        def add_count_row(raw_table_contents, table_contents):
            count_row = []
            for colid in range(len(raw_table_contents[0])):
                col_cells = [row[colid] for row in raw_table_contents]
                if len(col_cells) == len(set(col_cells)):
                    count_row.append([])
                else:
                    cell_cnt = defaultdict(int)
                    for cell in col_cells:
                        cell_cnt[cell] += 1
                    cells = sorted(cell_cnt, key=lambda cell: cell_cnt[cell] * -1)
                    cell_cnt_l = [f'{cell} {cell_cnt[cell]}' for cell in cells if cell_cnt[cell] > 1]
                    count_row.append(u'count :' + u', '.join(cell_cnt_l))
            table_contents.append([self.tokenizer_bert.tokenize(cell) if len(cell) else [] for cell in count_row])
            return table_contents
        
        contents = []
        raw_contents = []
        for ln in open(fnm).readlines():
            info = ln.strip("\n").split("#")
            raw_contents.append(info)
            info = [self.tokenizer_bert.tokenize(i) for i in info]
            contents.append(info)
        contents = add_count_row(raw_contents, contents)
        if len(set([len(i) for i in contents])) > 1:
            print([len(i) for i in contents])
        return contents,raw_contents


    def get_dat(self,dataset_path,which):
        if which == 'train':
            fnm = dataset_path+'/train_id.json'
        elif which == 'val':
            fnm = dataset_path+'/val_id.json'
        elif which == 'test':
            fnm = dataset_path+'/test_id.json'
        elif which == 'simple':
            fnm = dataset_path+'/simple_test_id.json'
        elif which == 'complex':
            fnm = dataset_path+'/complex_test_id.json'
        else:
            assert 1 == 2, "which should be in train/val/test"
        table_ids = eval(open(fnm).read())
        smps = []
        for tab_id in tqdm(table_ids):
            tab_dat = self.all_dat[tab_id]
            qs, labels, caption = tab_dat
            for q, label in zip(qs, labels):
                smp = [q, label, tab_id]
                smps.append(smp)
                if label==1 and which!='simple' and which!='complex':
                    self.verification_dataset.append({"claim":q,"tab_id":tab_id})
        print(f'{which} sample num={len(smps)}, pos_num={sum([i[1] for i in smps])}')
        return smps

    def trans_a_smp(self, smp):
        q, label, tab_id = smp
        q_tokenized = self.tokenizer_bert.tokenize(q)
        table_contents = self.tables[tab_id]
        q_tokenized = ["[CLS]"] + q_tokenized + ['[SEP]']
        token_ids = self.tokenizer_bert.convert_tokens_to_ids(q_tokenized)
        token_ids_source = [(-1, -1)] * len(q_tokenized)
        for rowid, content in enumerate(table_contents):
            for cell_id, cell in enumerate(content):
                token_ids.extend(self.tokenizer_bert.convert_tokens_to_ids(cell))
                token_ids_source.extend([(rowid, cell_id)] * len(cell))
        if len(token_ids) > LL:
            token_ids = token_ids[:LL]
            token_ids_source = token_ids_source[:LL]
        else:
            token_ids += [pad_token] * (LL - len(token_ids))
            token_ids_source += [(-2, -2)] * (LL - len(token_ids_source))

        if self.pretrain_mode == "tapas":
            table_contents_raw = self.tables_raw[tab_id]
            table_df = pd.DataFrame(table_contents_raw[1:], columns=table_contents_raw[0]).astype(str)
            inputs_tapas = self.tokenizer_tapas(table=table_df, queries=q, padding="max_length", truncation=True, return_tensors="pt")
            type_ids = inputs_tapas['token_type_ids']
        else:
            type1_len = len(q_tokenized)
            type_ids = [0] * type1_len + [1] * (LL - type1_len)

        return token_ids, type_ids, token_ids_source, label  # build_vmm(token_ids_source)


    def trans2ids(self, smps):
        res = []
        for smp in tqdm(smps):
            res.append(self.trans_a_smp(smp))
        return res

    def trans2ids_mlm(self, smps):
        res = []
        for smp in tqdm(smps):
            res.append(self.trans_a_smp_mlm(smp))
        return res

    def get_mlm_dat(self,dataset_path):
        fnms = [dataset_path+'/train_id.json',dataset_path+'/val_id.json',dataset_path+'/test_id.json']
        smps = []
        for fnm in fnms:
            table_ids = eval(open(fnm).read())
            for tab_id in tqdm(table_ids):
                tab_dat = self.all_dat[tab_id]
                qs, labels, caption = tab_dat
                for q, label in zip(qs, labels):
                    smp = [q, label, tab_id]
                    smps.append(smp)
        print(f'sample num={len(smps)}')
        return smps

    def trans_a_smp_mlm(self, smp):
        q, label, tab_id = smp
        q_tokenized = self.tokenizer_bert.tokenize(q)
        table_contents = self.tables[tab_id]
        q_tokenized = ["[CLS]"] + q_tokenized + ['[SEP]']
        token_ids = self.tokenizer_bert.convert_tokens_to_ids(q_tokenized)
        token_ids_source = [(-1, -1)] * len(q_tokenized)
        for rowid, content in enumerate(table_contents):
            for cell_id, cell in enumerate(content):
                token_ids.extend(self.tokenizer_bert.convert_tokens_to_ids(cell))
                token_ids_source.extend([(rowid, cell_id)] * len(cell))
        if len(token_ids) > LL:
            token_ids = token_ids[:LL]
            token_ids_source = token_ids_source[:LL]
        else:
            token_ids += [pad_token] * (LL - len(token_ids))
            token_ids_source += [(-2, -2)] * (LL - len(token_ids_source))
        batch = {"input_ids":[token_ids]}
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )

        if self.pretrain_mode == "tapas":
            table_contents_raw = self.tables_raw[tab_id]
            table_df = pd.DataFrame(table_contents_raw[1:], columns=table_contents_raw[0]).astype(str)
            inputs_tapas = self.tokenizer_tapas(table=table_df, queries=q, padding="max_length", truncation=True, return_tensors="pt")
            type_ids = inputs_tapas['token_type_ids']
        else:
            type1_len = len(q_tokenized)
            type_ids = [0] * type1_len + [1] * (LL - type1_len)

        return batch["input_ids"][0].tolist(), type_ids, token_ids_source, batch["labels"][0].tolist()  # build_vmm(token_ids_source)



    def load_all_the_dat(self,claim_path):
        all_dat = {}
        for fnm in [claim_path+"/r1_training_all.json", claim_path+"/r2_training_all.json"]:
            infos = eval(open(fnm).read())
            all_dat.update(infos)
            print(len(all_dat))
        return all_dat

    def mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs=  torch.tensor(inputs)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer_bert.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer_bert.convert_tokens_to_ids(self.tokenizer_bert.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer_bert), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


   

    def iter_batches(self, which="train", samples=None, batch_size=None):
        if which == 'train':
            return DataLoader(shuffle=True, dataset=self.train_dataset, batch_size=10, num_workers=3)
        if which == 'dev':
            return DataLoader(shuffle=False, dataset=self.dev_dataset, batch_size=10, num_workers=3)
        if which == 'test':
            return DataLoader(shuffle=False, dataset=self.test_dataset, batch_size=10, num_workers=3)
        if which == 'simple':
            return DataLoader(shuffle=False, dataset=self.simple_smps, batch_size=10, num_workers=3)
        if which == 'complex':
            return DataLoader(shuffle=False, dataset=self.complex_smps, batch_size=10, num_workers=3)

    def iter_batches_mlm(self, samples=None, batch_size=None):
        return DataLoader(shuffle=True, dataset=self.mlm_dataset, batch_size=10, num_workers=3)


class MyDataSet(Dataset):
    def __init__(self, smps, pretrain_mode):
        self.smps = smps
        self.pretrain_mode = pretrain_mode
        super().__init__()

    def __getitem__(self, i):
        token_ids, type_ids, token_ids_source, labels = self.smps[i]
        return np.asarray(token_ids), np.asarray(type_ids), np.asarray(build_vmm(token_ids_source, 'low')), np.asarray(
                build_vmm(token_ids_source, 'upper')),labels

    def __len__(self):
        return len(self.smps)

class MyDataSet_mlm(Dataset):
    def __init__(self, smps, pretrain_mode):
        self.smps = smps
        self.pretrain_mode = pretrain_mode
        super().__init__()

    def __getitem__(self, i):
        token_ids, type_ids, token_ids_source, labels = self.smps[i]
        return np.asarray(token_ids), np.asarray(type_ids), np.asarray(build_vmm(token_ids_source, 'low')), np.asarray(
            build_vmm(token_ids_source, 'upper')), np.asarray(labels)
    def __len__(self):
        return len(self.smps)


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

class TFVSP:
    def __init__(self, cuda_id):
        self.device = 'cuda:'+cuda_id
        self.dt = None
        self.pretrain_model = ""
        self.intermediate_model = ""
        self.finetune_model = ""
        

    def load_data(self, csv_path, claim_path, dataset_path, pretrain_mode, intermediate_mode):
        self.dt = DataManager(csv_path, claim_path, dataset_path, pretrain_mode, intermediate_mode)
    
    def fit_table_entailment(self, pretrain_mode, intermediate_mode, finetune_mode, output_path):
        
        #default: 
            #pretrain_mode = ['bert','tapas']
            #intermediate_mode = ['no','mlm','sat-mlm']
            #finetune_mode = ['sat-classification','classification']

        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(output_path+"/intermediate"):
            os.mkdir(output_path+"/intermediate")
        if not os.path.exists(output_path+"/finetune"):
            os.mkdir(output_path+"/finetune")

        self.pretrain_model = self.pretrain(output_path,pretrain_mode)
        
        if pretrain_mode == "bert":
            if intermediate_mode == 'no':
                self.intermediate_model= ""
            else:
                if intermediate_mode == 'sat-mlm':
                    self.intermediate_model = self.intermediate(use_structure="yes", output_intermediate_path=output_path+"/intermediate/Pbert",pretrain_mode=pretrain_mode)
                else:
                    self.intermediate_model = self.intermediate(use_structure="no", output_intermediate_path=output_path+"/intermediate/Pbert",pretrain_mode=pretrain_mode)

            if finetune_mode == 'sat-classification':
                self.finetune_model = self.finetune(use_structure="yes",pretrain_mode=pretrain_mode,intermediate_mode=intermediate_mode,
                    intermediate_model=self.intermediate_model,output_finetune_path=output_path+"/finetune/Pbert")
            else:
                self.finetune_model = self.finetune(use_structure="no",pretrain_mode=pretrain_mode,intermediate_mode=intermediate_mode,
                    intermediate_model=self.intermediate_model,output_finetune_path=output_path+"/finetune/Pbert")
        else:
            if intermediate_mode == 'no':
                self.intermediate_model= ""
            else:
                if intermediate_mode == 'sat-mlm':
                    self.intermediate_model = self.intermediate(use_structure="yes", output_intermediate_path=output_path+"/intermediate/Ptapas",pretrain_mode=pretrain_mode)
                else:
                    self.intermediate_model = self.intermediate(use_structure="no", output_intermediate_path=output_path+"/intermediate/Ptapas",pretrain_mode=pretrain_mode)

            if finetune_mode == 'sat-classification':
                self.finetune_model = self.finetune(use_structure="yes",pretrain_mode=pretrain_mode,intermediate_mode=intermediate_mode,
                    intermediate_model=self.intermediate_model,output_finetune_path=output_path+"/finetune/Ptapas")
            else:
                self.finetune_model = self.finetune(use_structure="no",pretrain_mode=pretrain_mode,intermediate_mode=intermediate_mode,
                    intermediate_model=self.intermediate_model,output_finetune_path=output_path+"/finetune/Ptapas")
        print("pretrain_model: "+self.pretrain_model)
        print("intermediate_model: "+self.intermediate_model)
        print("finetune_model: "+self.finetune_model)

    def pretrain(self, output_path, pretrain_mode):
        pretrain_model = ""
        if pretrain_mode=="bert":
            if not os.path.exists(output_path+"/intermediate/Pbert"):
                os.mkdir(output_path+"/intermediate/Pbert")
            if not os.path.exists(output_path+"/finetune/Pbert"):
                os.mkdir(output_path+"/finetune/Pbert")
            pretrain_model = 'bert-base-uncased'
        else:
            if not os.path.exists(output_path+"/intermediate/Ptapas"):
                os.mkdir(output_path+"/intermediate/Ptapas")
            if not os.path.exists(output_path+"/finetune/Ptapas"):
                os.mkdir(output_path+"/finetune/Ptapas")
            pretrain_model = 'google/tapas-base'
        return pretrain_model

    def intermediate(self, use_structure, output_intermediate_path,pretrain_mode):
        use_tapas="no"
        if pretrain_mode == 'tapas':
            use_tapas = "yes"
        return mlm_run(self.dt,self.device,use_structure,output_intermediate_path,use_tapas)
        
    
    def finetune(self,use_structure,pretrain_mode,intermediate_mode,intermediate_model,output_finetune_path):
        if intermediate_mode == "no":
            output_finetune_path = output_finetune_path+"/Ino"
        elif intermediate_mode == "mlm":
            output_finetune_path = output_finetune_path+"/Imlm"
        else:
            output_finetune_path = output_finetune_path+"/Isat"
        if not os.path.exists(output_finetune_path):
            os.mkdir(output_finetune_path)
        use_tapas="no"
        if pretrain_mode == 'tapas':
            use_tapas = "yes"
        return finetune_run(self.dt,self.device,use_structure,intermediate_model,output_finetune_path,use_tapas) 

    
    def score_table_entailment(self, pretrain_mode, finetune_mode, table_entailment_model_path):
        use_structure = "no"
        if finetune_mode == 'sat-classification':
            use_structure = "yes"
        use_tapas="no"
        if pretrain_mode == 'tapas':
            use_tapas = "yes"
        score_table_entailment(self.dt,self.device,use_structure,table_entailment_model_path,use_tapas) 


    def predict_table_entailment(self, claim, tab_id, pretrain_mode, finetune_mode, table_entailment_model_path):
        use_structure = "no"
        if finetune_mode == 'sat-classification':
            use_structure = "yes"
        use_tapas="no"
        if pretrain_mode == 'tapas':
            use_tapas = "yes"
        predict_table_entailment(self.dt,self.device,use_structure,claim,tab_id,table_entailment_model_path,use_tapas) 
    

    def score_verification(self, pretrain_mode, finetune_mode, table_entailment_model_path, k):
        use_structure = "no"
        if finetune_mode == 'sat-classification':
            use_structure = "yes"
        use_tapas="no"
        if pretrain_mode == 'tapas':
            use_tapas = "yes"
        score_verification(self.dt,self.device,use_structure,table_entailment_model_path,k,use_tapas) 


    def predict_verification(self, claim, pretrain_mode, finetune_mode, table_entailment_model_path,csv_path,k):
        use_structure = "no"
        if finetune_mode == 'sat-classification':
            use_structure = "yes"
        use_tapas="no"
        if pretrain_mode == 'tapas':
            use_tapas = "yes"
        predict_verification(self.dt,self.device,use_structure,claim,table_entailment_model_path,csv_path,k,use_tapas) 
    

if __name__ == '__main__':

    args = parser.parse_args()
    tfvsp = TFVSP(cuda_id=args.cuda_id)
   
    tfvsp.load_data(args.csv_path, args.claim_path, args.dataset_path, args.pretrain_mode, args.intermediate_mode,)
    #tfvsp.finetune(output_finetune_path="/home/cmk/fact/TFVSP/train_model",use_tapas="yes")
    

    tfvsp.fit_table_entailment(args.pretrain_mode, args.intermediate_mode, args.finetune_mode, args.output_path)
    #tfvsp.score_table_entailment(args.finetune_mode,tfvsp.finetune_model)

    '''
    finetune_model ="/home/cmk/fact/TFVSP/train_model/finetune/Pbert/Ino/Ftab/epoch_9_acc_0.6576920084040152.pt"
    tfvsp.score_table_entailment(args.pretrain_mode,args.finetune_mode,finetune_model)
    '''

    '''
    claim = "the highest number of winners from a previous round in the turkish cup was 54 in third round"
    tab_id = "2-1859269-1.html.csv"
    finetune_model ="/home/cmk/fact/TFVSP/train_model/finetune/Pbert/Ino/Ftab/epoch_9_acc_0.6576920084040152.pt"
    tfvsp.predict_table_entailment(claim,tab_id,args.pretrain_mode,args.finetune_mode,finetune_model)
    '''
    '''
    claim = "the highest number of winners from a previous round in the turkish cup was 54 in third round"
    finetune_model ="/home/cmk/fact/TFVSP/train_model/finetune/Pbert/Ino/Ftab/epoch_9_acc_0.6576920084040152.pt"
    tfvsp.predict_verification(claim,args.pretrain_mode,args.finetune_mode,finetune_model,args.csv_path,args.k)
    '''
    '''
    finetune_model ="/home/cmk/fact/TFVSP/train_model/finetune/Pbert/Ino/Ftab/epoch_9_acc_0.6576920084040152.pt"
    tfvsp.score_verification(args.pretrain_mode,args.finetune_mode,finetune_model,args.k)
    '''


    
    











