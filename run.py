# -*-codeing = utf-8 -*-
# @Time : 2023-12-1215:42
# @Author : 童宇
# @File : run.py
# @software :

import os
from utils.dataloader import bert_data
from model.mdfend import Trainer as MDFENDTrainer
from model.domain import Trainer as DOMAINTrainer

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']

        self.train_path = self.root_path + 'train-cut.csv'#如果9个领域就要改成train.csv
        self.val_path = self.root_path + 'val-cut.csv'#如果9个领域就要改成val.csv
        self.test_path = self.root_path + 'test-cut.csv'#如果9个领域就要改成test.csv


        #self.category_dict = {
        #        "经济": 0,
        #        "健康": 1,
        #        "军事": 2,
        #        "科学": 3,
        #        "政治": 4,
        #        "国际": 5,
        #        "教育": 6,
        #       "娱乐": 7,
        #      "社会": 8
        #}
        self.category_dict = {
               "经济": 0,
               "健康": 1,
               "军事": 2,
               "科学": 3,
               "政治": 4,
                "教育": 5,
                "娱乐": 6,
                "社会": 7
        }

    def get_dataloader(self):
        if self.emb_type == 'bert':
           loader = bert_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                              category_dict=self.category_dict, num_workers=self.num_workers)

        train_loader = loader.load_data(self.train_path,'data/train_cut_loader.pkl',True)
        val_loader = loader.load_data(self.val_path,'data/val_cut_loader.pkl',False)
        test_loader = loader.load_data(self.test_path,'data/test_cut_loader.pkl',False)
        # 如果9个领域就要改成train_loader.csv这种
        return train_loader, val_loader, test_loader

    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader()


        if self.model_name == 'mdfend':
            trainer = MDFENDTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name))
        if self.model_name == 'domain':
            trainer = DOMAINTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name))


        trainer.train()
