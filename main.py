# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset

from models.GDN_AR_Attention_TCN import GDN

from train import train
from test import val, test
# from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
# evaluate指标可以删除

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

        train, test = train_orig, test_orig

        if 'attack' in train.columns:
            print("attack in train.columns")
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)
        print("main_feature_map:", feature_map)  # 得到所有特征的名称list
        print("main_fc_struc:", fc_struc)  # 得到除自己之外的所有结点的候选集C

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        print("main_fc_edge_index", fc_edge_index)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        print("main_fc_edge_index", fc_edge_index)
        '''
        main_fc_edge_index tensor([[1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 5, 0, 1, 2, 3,
         5, 0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
         4, 5, 5, 5, 5, 5]])
        '''

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map)
        print("main_train_dataset_indata:", len(train_dataset_indata), len(train_dataset_indata[0]))  # 构建数据：没有label，将所有数据组合成list[6x43824]
        test_dataset_indata = construct_data(test, feature_map)
        print("main_test_dataset_indata:", len(test_dataset_indata), len(test_dataset_indata[0]))  # 构建数据：没有label，将所有数据组合成list[6x8784]

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, train_config['predict_length'], mode='train',
                                    config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, train_config['predict_length'], mode='test',
                                   config=cfg)

        # train_dataloader, val_dataloader = self.get_loaders(train_dataset, test_dataset, train_config['seed'], train_config['batch'], val_ratio=train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_scale1 = train_dataset.scale1
        self.train_scale_y = train_dataset.scale_y
        self.test_scale1 = test_dataset.scale1
        self.test_scale_y = test_dataset.scale_y

        self.train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch'], shuffle=False,
                                           num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0, drop_last=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'], shuffle=False, num_workers=0, drop_last=True)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map),
                         dim=train_config['dim'],
                         input_dim=train_config['slide_win'],
                         out_layer_num=train_config['out_layer_num'],
                         out_layer_inter_dim=train_config['out_layer_inter_dim'],
                         topk=train_config['topk']
                         ).to(self.device)


    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
            print("main_model_save_path_load_model_path:", model_save_path)
        else:
            model_save_path = self.get_save_path()[0]
            print("main_model_save_path:", model_save_path)
            print(self.model)
            nParams = sum([p.nelement() for p in self.model.parameters()])
            print('Number of model parameters is', nParams)
            self.train_log = train(self.model, model_save_path,
                                   config=train_config,
                                   train_dataloader=self.train_dataloader,
                                   val_dataloader=self.val_dataloader,
                                   feature_map=self.feature_map,
                                   test_dataloader=self.test_dataloader,
                                   test_dataset=self.test_dataset,
                                   train_dataset=self.train_dataset,
                                   dataset_name=self.env_config['dataset'],
                                   train_scale_y=self.train_scale_y,
                                   val_scale_y=self.test_scale_y
                                   )
            # print("main_self.train_log:", self.train_log)

        # test
        # self.model.load_state_dict(torch.load(model_save_path))
        self.model = torch.load(model_save_path)
        best_model = self.model.to(self.device)
        nParams = sum([p.nelement() for p in self.model.parameters()])
        print('Number of test model parameters is', nParams)

        self.train_result, train_rmse, train_mae, train_cc = test(best_model, self.train_dataloader, self.train_scale_y)
        self.val_result, val_rmse, val_mae, val_cc = test(best_model, self.val_dataloader, self.test_scale_y)
        self.test_result, test_rmse, test_mae, test_cc = test(best_model, self.test_dataloader, self.test_scale_y)
        # print("main_val_result: ", self.val_result)
        print(train_rmse, train_mae, train_cc)
        print(val_rmse, val_mae, val_cc)
        print(train_rmse.item(), train_mae.item(), train_cc.item(), val_rmse.item(), val_mae.item(), val_cc.item())

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            f'./Ablation_results/{dir_path}/best_{datestr}.pt',
            f'./Ablation_results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=32)
    parser.add_argument('-epoch', help='train epoch', type=int, default=30)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=1)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='All')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='Solar_hour')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=5)
    parser.add_argument('-comment', help='experiment comment', type=str, default='Solar_hour')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=128)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.2)
    parser.add_argument('-topk', help='topk num', type=int, default=3)  # 与每个特征最相关的topk个特征，5取得最好的效果
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('-slide_win', help='input_length', type=int, default=24)  # 输入数据长度
    parser.add_argument('-predict_length', help='predict length', type=int, default=24)  # 输出数据长度

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'predict_length': args.predict_length
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config, debug=False)
    main.run()

# CUDA_VISIBLE_DEVICES=1 python -u main.py |tee ./save_log/test
# CUDA_VISIBLE_DEVICES=4 python -u main.py |tee ./save_log/solar_96_96
# CUDA_VISIBLE_DEVICES=4 python -u main.py |tee ./save_log/TCN/solar_96_24_dilation1

# CUDA_VISIBLE_DEVICES=1 python -u main.py |tee ./save_log_7/GDN_only/batch_size_32/layer_1/lr_0.001/solar_24_24

# conda activate py37
# CUDA_VISIBLE_DEVICES=1 python -u main_GDN_AR.py -slide_win 24 -predict_length 24 -batch 32 -topk 3 |tee ./save_log_7/GDN_AR/batch_size_32/layer_1/lr_0.001/solar_24_24
# CUDA_VISIBLE_DEVICES=1 python -u main_GDN_AR.py |tee ./save_log_7/GDN_AR/batch_size_32/layer_1/lr_0.001/solar_48_24
