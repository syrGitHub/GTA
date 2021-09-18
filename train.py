import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, mean_squared_error, \
    mean_absolute_error
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')  # 可以改为sum试试结果

    return loss


def computecc(outputs, targets):
    """Computes and stores the average and current value"""
    # print("***************train_computecc", targets.shape, outputs.shape)  # torch.Size([64, 1, 4]) torch.Size([64, 1, 4])
    # print("train_computecc, outputs, targets", outputs, targets)
    xBar = targets.mean()
    yBar = outputs.mean()
    # print("train xBar, yBar", xBar, yBar)
    SSR = 0
    varX = 0  # 公式中分子部分
    varY = 0  # 公式中分母部分
    for i in range(0, targets.shape[0]):
        diffXXBar = targets[i] - xBar
        diffYYBar = outputs[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = torch.sqrt(varX * varY)
    xxx = SSR / SST
    # print("xxxxxxxxx", xxx)
    return torch.mean(xxx)


def rmse(preds, labels):
    loss = (preds - labels) ** 2
    #print("train_rmse_loss", loss)
    loss = torch.mean(loss)
    # print("train_rmse_loss", loss)
    return torch.sqrt(loss)


def mae(preds, labels):
    loss = torch.abs(preds - labels)
    # print("train_mae_loss", loss)
    # print("train_mae_torch.mean loss", torch.mean(loss))
    return torch.mean(loss)


def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None, feature_map={},
          test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None, train_scale_y=None,
          val_scale_y=None):
    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=config['decay'])

    now = time.time()

    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    acu_loss = 0
    min_loss = 1e+8

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        rmse_all = 0
        mae_all = 0
        t_train_predicted_list = []
        t_train_ground_list = []
        model.train()
        # for name in model.state_dict():
            # print(name)
        # print("before_train_model.state_dict()", model.state_dict()['embedding.weight'])

        j = 0
        for x, labels, edge_index in dataloader:
            model.train()
            # print("train_labels, edge_index:", labels, edge_index)
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]   # 原始的edge_index并没有用
            # print("train_labels, edge_index:", labels, edge_index)  # 输出batch_size个edge的堆叠

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)

            loss = loss_func(out, labels)

            # print("train_out.shape, labels.shape", out.shape, labels.shape)  # [6*96] torch.Size([32, 1])

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            # rmse_all += train_rmse.item()
            # mae_all += train_mae.item()

            if len(t_train_predicted_list) <= 0:
                t_train_predicted_list = out
                t_train_ground_list = labels
            else:
                t_train_predicted_list = torch.cat((t_train_predicted_list, out), dim=0)
                t_train_ground_list = torch.cat((t_train_ground_list, labels), dim=0)

            i += 1
            j += 1
        # print("train_j", j)

        # print("after_train_model.state_dict()", model.state_dict()['embedding.weight'])
        t_train_predicted_list = train_scale_y.inverse_transform(t_train_predicted_list)
        t_train_ground_list = train_scale_y.inverse_transform(t_train_ground_list)

        # print("train_t_train_predicted_list.shape, t_train_ground_list.shape", t_train_predicted_list.shape, t_train_ground_list.shape)
        rmse_all = rmse(t_train_predicted_list, t_train_ground_list)
        mae_all = mae(t_train_predicted_list, t_train_ground_list)
        train_cc = computecc(t_train_predicted_list, t_train_ground_list)

        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f}, RMSE:{:.8f}, MAE:{:.8f}, CC:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss / len(dataloader), acu_loss, rmse_all, mae_all, train_cc),
            flush=True
        )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result, val_rmse, val_mae, val_cc = val(model, val_dataloader, val_scale_y)

            print('val : (Loss:{:.8f},  RMSE:{:.8f}, MAE:{:.8f}, CC:{:.8f})'.format(val_loss, val_rmse, val_mae, val_cc),
                  flush=True)

            if val_loss < min_loss:
                # torch.save(model.state_dict(), save_path)
                torch.save(model, save_path)
                print("save best model at ", save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    return train_loss_list
