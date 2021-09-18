import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

from util.data import *
from util.preprocess import *


def computecc(outputs, targets):
    """Computes and stores the average and current value"""
    # print("***************test_computecc", targets.shape, outputs.shape)  # torch.Size([64, 1, 4]) torch.Size([64, 1, 4])
    # print("test_computecc, outputs, targets", outputs, targets, targets.shape, outputs.shape)  # torch.Size([8665, 1]) torch.Size([8665, 1])
    xBar = targets.mean()
    yBar = outputs.mean()
    # print(xBar,yBar)
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
    # print(loss, loss.shape)
    loss = torch.mean(loss)
    # print(loss, loss.shape)
    return torch.sqrt(loss)


def mae(preds, labels):
    loss = torch.abs(preds - labels)
    # print("loss", loss, loss.shape)
    # print("return", torch.mean(loss))
    return torch.mean(loss)


def val(model, dataloader, test_scale):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []

    t_test_predicted_list = []
    t_test_ground_list = []

    test_len = len(dataloader)

    model.eval()
    # print("before_val_model.state_dict()", model.state_dict()['embedding.weight'])
    # nParams = sum([p.nelement() for p in model.parameters()])
    # print('Number of val model parameters is', nParams)

    i = 0
    acu_loss = 0
    val_predicted = []

    for x, y, edge_index in dataloader:
        x, y, edge_index = [item.to(device).float() for item in [x, y, edge_index]]

        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)

            loss = loss_func(predicted, y)

            # print("test_x, predicted, y", x, predicted, y)

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        # val_rmse = rmse_all.item()
        # val_mae = mae_all.item()

        i += 1
        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    # print("after_val_model.state_dict()", model.state_dict()['embedding.weight'])
    t_test_predicted_list = test_scale.inverse_transform(t_test_predicted_list)
    t_test_ground_list = test_scale.inverse_transform(t_test_ground_list)
    val_cc = computecc(t_test_predicted_list, t_test_ground_list)
    val_rmse = rmse(t_test_predicted_list, t_test_ground_list)
    val_mae = mae(t_test_predicted_list, t_test_ground_list)
    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list], val_rmse, val_mae, val_cc


def test(model, dataloader, test_scale):
    # test data
    device = get_device()
    t_test_predicted_list = []
    t_test_ground_list = []

    for x, y, edge_index in dataloader:
        x, y, edge_index = [item.to(device).float() for item in [x, y, edge_index]]
        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
        if len(t_test_predicted_list) <= 0:
            t_test_predicted_list = predicted
            t_test_ground_list = y
        else:
            t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
            t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)

    t_test_predicted_list = test_scale.inverse_transform(t_test_predicted_list)
    t_test_ground_list = test_scale.inverse_transform(t_test_ground_list)
    val_cc = computecc(t_test_predicted_list, t_test_ground_list)
    val_rmse = rmse(t_test_predicted_list, t_test_ground_list)
    val_mae = mae(t_test_predicted_list, t_test_ground_list)
    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()

    return [test_predicted_list, test_ground_list], val_rmse, val_mae, val_cc
