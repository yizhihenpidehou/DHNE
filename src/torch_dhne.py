import datetime

import numpy as np
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()
import argparse
from functools import reduce
import math
import time
from sklearn.metrics import roc_auc_score

from dataset import read_data_sets, embedding_lookup

parser = argparse.ArgumentParser("hyper-network embedding", fromfile_prefix_chars='@')
parser.add_argument('--data_path', type=str, help='Directory to load data.')
parser.add_argument('--save_path', type=str, help='Directory to save data.')
parser.add_argument('-s', '--embedding_size', type=int, nargs=3, default=[32, 32, 32],
                    help='The embedding dimension size')
parser.add_argument('--prefix_path', type=str, default='model', help='.')
parser.add_argument('--hidden_size', type=int, default=64, help='The hidden full connected layer size')
parser.add_argument('-e', '--epochs_to_train', type=int, default=10,
                    help='Number of epoch to train. Each epoch processes the training data once completely')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Number of training examples processed per step')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('-a', '--alpha', type=float, default=1, help='radio of autoencoder loss')
parser.add_argument('-neg', '--num_neg_samples', type=int, default=5, help='Neggative samples per training example')
parser.add_argument('-o', '--options', type=str, help='options files to read, if empty, stdin is used')
parser.add_argument('--seed', type=int, help='random seed')

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DHNE(nn.Module):
    def __init__(self, dim_feature, embedding_size, hidden_size):
        super(DHNE, self).__init__()
        self.dim_feature = dim_feature
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # self.inputs = nn.Sequential()
        self.encode0 = nn.Sequential(
            nn.Linear(in_features=self.dim_feature[0], out_features=self.embedding_size[0])
        )
        self.encode1 = nn.Sequential(
            nn.Linear(in_features=self.dim_feature[1], out_features=self.embedding_size[1])
        )
        self.encode2 = nn.Sequential(
            nn.Linear(in_features=self.dim_feature[2], out_features=self.embedding_size[2])
        )
        self.decode_layer0 = nn.Linear(in_features=self.embedding_size[0], out_features=self.dim_feature[0])
        self.decode_layer1 = nn.Linear(in_features=self.embedding_size[1], out_features=self.dim_feature[1])
        self.decode_layer2 = nn.Linear(in_features=self.embedding_size[2], out_features=self.dim_feature[2])

        self.hidden_layer = nn.Linear(in_features=sum(self.embedding_size), out_features=self.hidden_size)
        self.ouput_layer = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, input0, input1, input2):
        input0 = self.encode0(input0)
        input0 = torch.tanh(input0)
        decode0 = self.decode_layer0(input0)
        decode0 = torch.sigmoid(decode0)

        input1 = self.encode1(input1)
        input1 = torch.tanh(input1)
        decode1 = self.decode_layer1(input1)
        decode1 = torch.sigmoid(decode1)

        input2 = self.encode2(input2)
        input2 = torch.tanh(input2)
        decode2 = self.decode_layer2(input2)
        decode2 = torch.sigmoid(decode2)

        merged = torch.tanh(torch.cat((input0, input1, input2), dim=1))
        merged = self.hidden_layer(merged)
        merged = self.ouput_layer(merged)
        merged = torch.sigmoid(merged)
        return [decode0, decode1, decode2, merged]


def sparse_autoencoder_error(y_true, y_pred):
    return torch.mean(
        torch.square(torch.mul(torch.sign(y_true), torch.Tensor((y_true.detach().numpy() - y_pred))))).detach().numpy()


def train(model, optimizer, train_data, test_data, alpha=1.0):
    loss1 = []
    for i in range(0, 3):
        loss1.append(sparse_autoencoder_error(train_data[i], test_data[i]))
    # print("loss1:",loss1)
    train_data3 = train_data[3]
    test_data3 = torch.tensor(test_data[3]).reshape(16, 1)
    loss2 = nn.functional.binary_cross_entropy(train_data3, test_data3.float())
    # print("loss1:",loss1,"loss2:",loss2)

    optimizer.zero_grad()
    loss2.backward()
    optimizer.step()


def load_config(config_file):
    with open(config_file, 'r') as f:
        args = parser.parse_args(reduce(lambda a, b: a + b, map(lambda x: ('--' + x).strip().split(), f.readlines())))
    return args


if __name__ == '__main__':
    args = parser.parse_args()
    if args.options is not None:
        args = load_config(args.options)
    if args.seed is not None:
        np.random.seed(args.seed)
    #    读数据集,构建超图
    dataset = read_data_sets(args.data_path)
    # 训练集与测试集
    train_dataset = dataset.train
    test_dataset = dataset.test
    # 设置特征维度
    args.dim_feature = [sum(dataset.train.nums_type) - n for n in dataset.train.nums_type]
    print("dataset.train.nums_type:", dataset.train.nums_type)
    print("dataset.train.nums_type:", dataset.train.nums_type)
    print("args.dim_feature:", args.dim_feature)
    #

    begin = time.time()
    # 训练
    m = DHNE(dim_feature=args.dim_feature, embedding_size=[args.embedding_size[0], args.embedding_size[1], args.embedding_size[2]], hidden_size=args.hidden_size)
    # 定义优化器
    optimizer = torch.optim.RMSprop(params=m.parameters(), lr=args.learning_rate)

    m = m.to(device)

    # 训练的轮数
    epoch = args.epochs_to_train
    total_avg_auc = 0
    start_time = time.time()
    print("dataset_len:", train_dataset.nums_examples)
    for i in range(epoch):

        print("--------第 {} 轮训练开始--------".format(i + 1))
        m.train()
        # 训练步骤开始
        train_dataset.index_in_epoch = 0

        gen = train_dataset.next_batch2(dataset.embeddings, batch_size=args.batch_size, num_neg_samples=args.num_neg_samples)
        while gen is not None:
            gen = list(gen)
            decode0, decode1, decode2, classify_res = m(torch.Tensor(gen[0]['input_0']),
                                                        torch.Tensor(gen[0]['input_1']),
                                                        torch.Tensor(gen[0]['input_2']))
            target_d0, target_d1, target_d2, target_classify_res = gen[1]["decode_0"], gen[1]["decode_1"], gen[1][
                "decode_2"], gen[1]["classify_layer"]
            # 训练过程中的输入与输出
            train_data = [decode0, decode1, decode2, classify_res]
            test_data = [target_d0, target_d1, target_d2, target_classify_res]
            # print("classify_res:", classify_res, " target:", target_classify_res)
            loss1 = []
            # 获得decoder的loss
            for j in range(0, 3):
                loss1.append(sparse_autoencoder_error(train_data[j], test_data[j]))
            # print("loss1:",loss1)

            train_data3 = train_data[3]
            test_data3 = torch.tensor(test_data[3]).reshape(train_data[3].size(dim=0), 1)
            # 或者分类层的loss
            loss2 = nn.functional.binary_cross_entropy(train_data3, test_data3.float()) + args.alpha * (sum(loss1))
            # print("loss2:",loss2)
            # 优化器调优
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            # if j % 100 == 0:
            loss = loss2.item()
            # print(f"loss: {loss:>7f}")
            gen = train_dataset.next_batch2(dataset.embeddings, batch_size=args.batch_size, num_neg_samples=args.num_neg_samples)

        print("--------第 {} 轮验证开始--------".format(i + 1))
        m.eval()
        roc_auc_score_avg = 0
        with torch.no_grad():
            test_dataset.index_in_epoch = 0
            print("dataset_len:", test_dataset.nums_examples)
            gen = test_dataset.next_batch2(dataset.embeddings, batch_size=args.batch_size, num_neg_samples=args.num_neg_samples)
            test_num = 0
            while gen is not None:
                test_num+=1
                gen = list(gen)
                decode0, decode1, decode2, classify_res = m(torch.Tensor(gen[0]['input_0']),
                                                            torch.Tensor(gen[0]['input_1']),
                                                            torch.Tensor(gen[0]['input_2']))
                target_d0, target_d1, target_d2, target_classify_res = gen[1]["decode_0"], gen[1]["decode_1"], \
                                                                       gen[1][
                                                                           "decode_2"], gen[1]["classify_layer"]


                # 训练过程中的输入与输出
                train_data = [decode0, decode1, decode2, classify_res]
                test_data = [target_d0, target_d1, target_d2, target_classify_res]

                loss1 = []
                # 获得decoder的loss
                for j in range(0, 3):
                    loss1.append(sparse_autoencoder_error(train_data[j], test_data[j]))
                train_data3 = train_data[3]
                test_data3 = torch.tensor(test_data[3]).reshape(train_data[3].size(dim=0), 1)
                # 或者分类层的loss
                loss2 = nn.functional.binary_cross_entropy(train_data3, test_data3.float()) + args.alpha * sum(
                    loss1)

                loss = loss2.item()
                print(f"test_loss: {loss:>7f}")
                gen = test_dataset.next_batch2(dataset.embeddings, batch_size=args.batch_size)
                # print("train_data[3]:",train_data[3].data.numpy().reshape(1,train_data[3].size(dim=0))[0])
                # print("test_data[3]:",test_data[3])
                roc_auc_score_avg+=roc_auc_score(np.array(test_data[3]), np.array(
                    train_data[3].data.numpy().reshape(1, train_data[3].size(dim=0))[0]))
                print("roc_auc_score:", roc_auc_score(np.array(test_data[3]), np.array(
                    train_data[3].data.numpy().reshape(1, train_data[3].size(dim=0))[0])))
        roc_auc_score_avg/=test_num
        total_avg_auc+=roc_auc_score_avg
        print("roc_auc_score_avg:",roc_auc_score_avg)
    end = time.time()
    print("time, ", end - begin)
    print("total_avg_auc:",total_avg_auc/epoch)
    # import numpy as np

    # embedding = np.load('/Users/yizhihenpidehou/Desktop/fdu/eg/DHNE/result/GPS/model_16/embeddings.npy',allow_pickle=True)

    # 保存模型
    # h.save()
    # 保存embedding
    # h.save_embeddings(dataset)
    # 清除模型缓存
    # K.clear_session()
