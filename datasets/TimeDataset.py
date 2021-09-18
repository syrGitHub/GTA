import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
import numpy as np


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler():
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min)) + self.min


def normalized(x, y, normalize):
    print("preprocess_m,n:", x.shape, y.shape)  # torch.Size([43705, 6, 96]) torch.Size([43705, 1])
    '''
    if (normalize == 0):
        # print("000000000util: _normalized")   # 不进行归一化
        dataset = dataset

    if (normalize == 1):
        # print("1111111111util: _normalized")
        dataset = dataset / np.max(dataset)
    '''
    # normlized by the maximum value of each row(sensor).
    if (normalize == 2):  # 最大最小归一化
        # dataset = np.array(dataset)
        scale = []
        scaler1 = MinMaxScaler(min=x[:, 0, :].min(), max=x[:, 0, :].max())
        scaler2 = MinMaxScaler(min=x[:, 1, :].min(), max=x[:, 1, :].max())
        scaler3 = MinMaxScaler(min=x[:, 2, :].min(), max=x[:, 2, :].max())
        scaler4 = MinMaxScaler(min=x[:, 3, :].min(), max=x[:, 3, :].max())
        scaler5 = MinMaxScaler(min=x[:, 4, :].min(), max=x[:, 4, :].max())
        scaler6 = MinMaxScaler(min=x[:, 5, :].min(), max=x[:, 5, :].max())
        scaler7 = MinMaxScaler(min=x[:, 6, :].min(), max=x[:, 6, :].max())
        scaler_y = MinMaxScaler(min=y.min(), max=y.max())
        x[:, 0, :] = scaler1.transform(x[:, 0, :])
        x[:, 1, :] = scaler2.transform(x[:, 1, :])
        x[:, 2, :] = scaler3.transform(x[:, 2, :])
        x[:, 3, :] = scaler4.transform(x[:, 3, :])
        x[:, 4, :] = scaler5.transform(x[:, 4, :])
        x[:, 5, :] = scaler6.transform(x[:, 5, :])
        x[:, 6, :] = scaler7.transform(x[:, 6, :])
        y = scaler_y.transform(y)
        # print("TimeDataset_tansform_x.shape, x, y.shape, y", x.shape, x, y.shape, y)  # torch.Size([43705, 6, 96]) torch.Size([43705, 1])
        print("TimeDataset_tansform_x.shape, x, y.shape, y", x.shape, y.shape)  # torch.Size([43705, 6, 96]) torch.Size([43705, 1])
        scale.append(scaler1)
        scale.append(scaler2)
        scale.append(scaler3)
        scale.append(scaler4)
        scale.append(scaler5)
        scale.append(scaler6)
        scale.append(scaler7)

    if (normalize == 3):  # 标准差方差归一化
        # dataset = np.array(dataset)
        scale = []
        scaler1 = StandardScaler(mean=x[:, 0, :].mean(), std=x[:, 0, :].std())
        scaler2 = StandardScaler(mean=x[:, 1, :].mean(), std=x[:, 1, :].std())
        scaler3 = StandardScaler(mean=x[:, 2, :].mean(), std=x[:, 2, :].std())
        scaler4 = StandardScaler(mean=x[:, 3, :].mean(), std=x[:, 3, :].std())
        scaler5 = StandardScaler(mean=x[:, 4, :].mean(), std=x[:, 4, :].std())
        scaler6 = StandardScaler(mean=x[:, 5, :].mean(), std=x[:, 5, :].std())
        scaler7 = StandardScaler(mean=x[:, 6, :].mean(), std=x[:, 6, :].std())
        scaler_y = StandardScaler(mean=y.mean(), std=y.std())
        x[:, 0, :] = scaler1.transform(x[:, 0, :])
        x[:, 1, :] = scaler2.transform(x[:, 1, :])
        x[:, 2, :] = scaler3.transform(x[:, 2, :])
        x[:, 3, :] = scaler4.transform(x[:, 3, :])
        x[:, 4, :] = scaler5.transform(x[:, 4, :])
        x[:, 5, :] = scaler6.transform(x[:, 5, :])
        x[:, 6, :] = scaler6.transform(x[:, 6, :])
        y = scaler_y.transform(y)
        # print("TimeDataset_tansform_x.shape, x, y.shape, y", x.shape, x, y.shape, y)
        print("TimeDataset_tansform_x.shape, x, y.shape, y", x.shape, y.shape)
        scale.append(scaler1)
        scale.append(scaler2)
        scale.append(scaler3)
        scale.append(scaler4)
        scale.append(scaler5)
        scale.append(scaler6)
        scale.append(scaler7)
    return x, y, scaler1, scaler_y


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, predict_length, mode='train', config=None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode
        self.predict_length = predict_length

        x_data = raw_data

        data = x_data

        # to tensor
        data = torch.tensor(data).double()

        self.x, self.y, self.scale1, self.scale_y = self.process(data)

    def __len__(self):
        return len(self.x)

    def process(self, data):
        x_arr, y_arr = [], []
        predict_length = self.predict_length
        print("TimeDataset_predict_length:", predict_length)

        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']  # 5,1
                                   ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape
        print("TimeDataset_data.shape: ", data.shape)  # [6,43824]

        rang = range(slide_win, total_time_len - predict_length + 1, slide_stride) \
            if is_train else range(slide_win, total_time_len - predict_length + 1)
        print("TimeDataset_rang, is_train: ", rang, is_train)  # range(5, 43824) True

        for i in rang:
            ft = data[:, i - slide_win:i]
            tar = data[0, i + predict_length - 1]

            x_arr.append(ft)
            y_arr.append(tar)

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        y = y.reshape(y.shape[0], 1)
        # print("TimeDataset_x_arr.shape, y_arr.shape, x_arr, y_arr:", x.shape, y.shape, x, y)    # torch.Size([43705, 6, 96]) torch.Size([43705, 1])
        x, y, scale1, scale_y = normalized(x, y, 2)
        # print("TimeDataset_x,y:", x.shape, y.shape)  # ([43705, 6, 96]) torch.Size([43705, 1])

        return x, y, scale1, scale_y

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()

        # print("TimeDataset_feature, y:", feature, y)    # feature 是 x, y是预测标签
        edge_index = self.edge_index.long()
        # print("TimeDataset_edge_index:", edge_index)
        '''TimeDataset_edge_index: tensor([[1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 1, 3, 4, 5, 0, 1, 2, 4, 5, 0, 1, 2, 3,
         5, 0, 1, 2, 3, 4],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
         4, 5, 5, 5, 5, 5]])'''

        return feature, y, edge_index
