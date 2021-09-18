# Writing h5

import pandas as pd
dataset = pd.read_csv("/home/sunyanru19s/pytorch/GDN-main/data/Solar_hour/train.txt", sep=',', header=0)
print(dataset)
dataset.to_csv("/home/sunyanru19s/pytorch/GDN-main/data/Solar_hour/train.csv", index=False)

dataset1 = pd.read_csv("/home/sunyanru19s/pytorch/GDN-main/data/Solar_hour/test.txt", sep=',', header=0)
print(dataset1)
dataset1.to_csv("/home/sunyanru19s/pytorch/GDN-main/data/Solar_hour/test.csv", index=False)


# 训练集：2011-2015 测试集：2016年