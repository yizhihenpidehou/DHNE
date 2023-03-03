import numpy as np
import os
import copy
import collections
import scipy.io as sio
import operator
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
import sys
import itertools
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )
#
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )
#
# train_dataloader = DataLoader(training_data, batch_size=64)
# test_dataloader = DataLoader(test_data, batch_size=64)
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'embeddings', 'node_cluster',
                                               'labels', 'idx_label', 'label_name'])


class DataSet(object):

    def __init__(self, edge, nums_type, **kwargs):
        self.edge = edge
        self.edge_set = set(map(tuple, edge))
        self.nums_type = nums_type
        self.kwargs = kwargs
        self.nums_examples = len(edge)
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, embeddings, batch_size=16, num_neg_samples=1, pair_radio=0.9, sparse_input=True):

        """
            Return the next `batch_size` examples from this data set.
            if num_neg_samples = 0, there is no negative sampling.
        """
        while self.index_in_epoch < self.nums_examples:
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
            if self.index_in_epoch > self.nums_examples:
                self.epochs_completed += 1
                np.random.shuffle(self.edge)
                start = 0
                self.index_in_epoch = batch_size
                assert self.index_in_epoch <= self.nums_examples
            end = self.index_in_epoch
            neg_data = []
            for i in range(start, end):
                n_neg = 0
                while (n_neg < num_neg_samples):
                    ### warning !!! we need deepcopy to copy list
                    index = copy.deepcopy(self.edge[i])
                    mode = np.random.rand()
                    if mode < pair_radio:
                        type_ = np.random.randint(3)
                        node = np.random.randint(self.nums_type[type_])
                        index[type_] = node
                    else:
                        types_ = np.random.choice(3, 2, replace=False)
                        node_1 = np.random.randint(self.nums_type[types_[0]])
                        node_2 = np.random.randint(self.nums_type[types_[1]])
                        index[types_[0]] = node_1
                        index[types_[1]] = node_2
                    if tuple(index) in self.edge_set:
                        continue
                    n_neg += 1
                    neg_data.append(index)
            if len(neg_data) > 0:
                batch_data = np.vstack((self.edge[start:end], neg_data))
                nums_batch = len(batch_data)
                labels = np.zeros(nums_batch)
                labels[0:end - start] = 1
                perm = np.random.permutation(nums_batch)
                batch_data = batch_data[perm]
                labels = labels[perm]
            else:
                batch_data = self.edge[start:end]
                nums_batch = len(batch_data)
                labels = np.ones(len(batch_data))
            batch_e = embedding_lookup(embeddings, batch_data, sparse_input)
            # print("batch_data:",dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]))
            return (dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]),
                    dict([('decode_{}'.format(i), batch_e[i]) for i in range(3)] + [('classify_layer', labels)]))



def embedding_lookup(embeddings, index, sparse_input=True):
    if sparse_input:
        return [embeddings[i][index[:, i], :].todense() for i in range(3)]
    else:
        return [embeddings[i][index[:, i], :] for i in range(3)]


def read_data_sets(train_dir):
    TRAIN_FILE = 'train_data.npz'
    TEST_FILE = 'test_data.npz'
    data = np.load(os.path.join(train_dir, TRAIN_FILE))
    train_data = DataSet(data['train_data'], data['nums_type'])
    labels = data['labels'] if 'labels' in data else None
    idx_label = data['idx_label'] if 'idx_label' in data else None
    label_set = data['label_name'] if 'label_name' in data else None
    del data
    data = np.load(os.path.join(train_dir, TEST_FILE))
    test_data = DataSet(data['test_data'], data['nums_type'])
    node_cluster = data['node_cluster'] if 'node_cluster' in data else None
    test_labels = data['labels'] if 'labels' in data else None
    del data
    # 初始embedding由超边生成
    embeddings = generate_embeddings(train_data.edge, train_data.nums_type)
    return Datasets(train=train_data, test=test_data, embeddings=embeddings, node_cluster=node_cluster,
                    labels=labels, idx_label=idx_label, label_name=label_set)


def generate_H(edge, nums_type):
    nums_examples = len(edge)
    H = [csr_matrix((np.ones(nums_examples), (edge[:, i], range(nums_examples))), shape=(nums_type[i], nums_examples))
         for i in range(3)]
    return H


def dense_to_onehot(labels):
    return np.array(map(lambda x: [x * 0.5 + 0.5, x * -0.5 + 0.5], list(labels)), dtype=float)


def generate_embeddings(edge, nums_type, H=None):
    if H is None:
        H = generate_H(edge, nums_type)
    # print("nums_type:",nums_type)
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype('float') for i in range(3)]
    '''

    [<146x75 sparse matrix of type '<class 'numpy.float64'>'
	with 1199 stored elements in Compressed Sparse Row format>, <70x151 sparse matrix of type '<class 'numpy.float64'>'
	with 834 stored elements in Compressed Sparse Row format>, <5x216 sparse matrix of type '<class 'numpy.float64'>'
	with 629 stored elements in Compressed Sparse Row format>]

    '''
    print("embedding:", embeddings)
    ### 0-1 scaling
    for i in range(3):
        # min(1)返回该矩阵中每一行的最小值
        # min(0)返回该矩阵中每一列的最小值
        # todense() 返回密集表示
        # 展平矩阵，将多维矩阵展平为一维矩阵
        # embedding[i] 是一个三元组
        # flatten 将多维矩阵展开成一维
        col_max = np.array(embeddings[i].max(0).todense()).flatten()
        # print("col_max:",col_max)
        # print("i:",i,"embedding:",embeddings[i])
        # 用于得到数组array中非零元素的位置（数组索引）的函数
        _, col_index = embeddings[i].nonzero()
        embeddings[i].data /= col_max[col_index]
    return embeddings

# read_data_sets("../data/GPS")
