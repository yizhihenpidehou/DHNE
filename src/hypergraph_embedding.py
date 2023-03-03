import numpy as np
import os, sys
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
from functools import reduce
import math
import time
import torchmetrics
from keras.models import Model
from keras import regularizers, optimizers
from keras.layers import Input, Dense, concatenate
from keras import backend as K
from keras.models import load_model

from dataset import read_data_sets, embedding_lookup




parser = argparse.ArgumentParser("hyper-network embedding", fromfile_prefix_chars='@')
parser.add_argument('--data_path', type=str, help='Directory to load data.')
parser.add_argument('--save_path', type=str, help='Directory to save data.')
parser.add_argument('-s', '--embedding_size', type=int, nargs=3, default=[32, 32, 32], help='The embedding dimension size')
parser.add_argument('--prefix_path', type=str, default='model', help='.')
parser.add_argument('--hidden_size', type=int, default=64, help='The hidden full connected layer size')
parser.add_argument('-e', '--epochs_to_train', type=int, default=10, help='Number of epoch to train. Each epoch processes the training data once completely')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Number of training examples processed per step')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='initial learning rate')
parser.add_argument('-a', '--alpha', type=float, default=1, help='radio of autoencoder loss')
parser.add_argument('-neg', '--num_neg_samples', type=int, default=5, help='Neggative samples per training example')
parser.add_argument('-o', '--options', type=str, help='options files to read, if empty, stdin is used')
parser.add_argument('--seed', type=int, help='random seed')


class hypergraph(object):
    def __init__(self, options):
        self.options=options
        self.build_model()

    def sparse_autoencoder_error(self, y_true, y_pred):
        # print("y_true:", y_true.shape)
        print("loss:", K.mean(K.square(K.sign(y_true)*(y_true-y_pred)), axis=-1))
        return K.mean(K.square(K.sign(y_true)*(y_true-y_pred)), axis=-1)


    def build_model(self):
        ### TO DO: tensorflow supports sparse_placeholder and sparse_matmul from version 1.4
        # 三种类型的节点，且每条超边为3个节点
        self.inputs = [Input(shape=(self.options.dim_feature[i], ), name='input_{}'.format(i), dtype='float') for i in range(3)]

        ### auto-encoder
        self.encodeds = [Dense(self.options.embedding_size[i], activation='tanh', name='encode_{}'.format(i))(self.inputs[i]) for i in range(3)]
        self.decodeds = [Dense(self.options.dim_feature[i], activation='sigmoid', name='decode_{}'.format(i),
                        activity_regularizer = regularizers.l2(0.0))(self.encodeds[i]) for i in range(3)]

        self.merged = concatenate(self.encodeds, axis=1)
        # tanh:non-linear activation
        self.hidden_layer = Dense(self.options.hidden_size, activation='tanh', name='full_connected_layer')(self.merged)
        self.ouput_layer = Dense(1, activation='sigmoid', name='classify_layer')(self.hidden_layer)
        print("self.decodeds+[self.ouput_layer]:",self.decodeds+[self.ouput_layer])
        self.model = Model(inputs=self.inputs, outputs=self.decodeds+[self.ouput_layer])

        self.model.compile(optimizer=optimizers.RMSprop(lr=self.options.learning_rate),
                loss=[self.sparse_autoencoder_error]*3+['binary_crossentropy'],
                              loss_weights=[self.options.alpha]*3+[1.0],
                              metrics=dict([('decode_{}'.format(i), 'mse') for i in range(3)]+[('classify_layer', 'accuracy')]))
        # m = model()
        # print("h:", h.shape)
        self.model.summary()

    def train(self, dataset):
        self.hist = self.model.fit_generator(
                dataset.train.next_batch(dataset.embeddings, self.options.batch_size, num_neg_samples=self.options.num_neg_samples),
                validation_data=dataset.test.next_batch(dataset.embeddings, self.options.batch_size, num_neg_samples=self.options.num_neg_samples),
                validation_steps=1,
                steps_per_epoch=math.ceil(dataset.train.nums_examples/self.options.batch_size),
                epochs=self.options.epochs_to_train, verbose=1)

    def predict(self, embeddings, data):
        test = embedding_lookup(embeddings, data)
        return self.model.predict(test, batch_size=self.options.batch_size)[3]

    def fill_feed_dict(self, embeddings, nums_type, x, y):
        batch_e = embedding_lookup(embeddings, x)
        return (dict([('input_{}'.format(i), batch_e[i]) for i in range(3)]),
                dict([('decode_{}'.format(i), batch_e[i]) for i in range(3)]+[('classify_layer', y)]))
        # return res

    def get_embeddings(self, dataset):
        shift = np.append([0], np.cumsum(dataset.train.nums_type))
        embeddings = []
        for i in range(3):
            index = range(dataset.train.nums_type[i])
            batch_num = math.ceil(1.0*len(index)/self.options.batch_size)
            ls = np.array_split(index, batch_num)
            ps = []
            for j, lss in enumerate(ls):
                embed = K.get_session().run(self.encodeds[i], feed_dict={
                    self.inputs[i]: dataset.embeddings[i][lss, :].todense()})
                ps.append(embed)
            ps = np.vstack(ps)
            embeddings.append(ps)
        return embeddings

    def save(self):
        prefix = '{}_{}'.format(self.options.prefix_path, self.options.embedding_size[0])
        prefix_path = os.path.join(self.options.save_path, prefix)
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        self.model.save(os.path.join(prefix_path, 'model.h5'))
        with open(os.path.join(prefix_path, 'config.txt'), 'w') as f:
            for key, value in vars(self.options).items():
                if value is None:
                    continue
                if type(value) == list:
                    s_v = " ".join(list(map(str, value)))
                else:
                    s_v = str(value)
                f.write(key+" "+s_v+'\n')

    def save_embeddings(self, dataset, file_name='embeddings.npy'):
        emds = self.get_embeddings(dataset)
        prefix = '{}_{}'.format(self.options.prefix_path, self.options.embedding_size[0])
        prefix_path = os.path.join(self.options.save_path, prefix)
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)
        np.save(open(os.path.join(prefix_path, file_name), 'wb'), emds)

    def load(self):
        prefix_path = os.path.join(self.options.save_path, '{}_{}'.format(self.options.prefix_path, self.options.embedding_size[0]))
        self.model = load_model(os.path.join(prefix_path, 'model.h5'), custom_objects={'sparse_autoencoder_error': self.sparse_autoencoder_error})

def load_config(config_file):
    with open(config_file, 'r') as f:
        args = parser.parse_args(reduce(lambda a, b: a+b, map(lambda x: ('--'+x).strip().split(), f.readlines())))
    return args

def load_hypergraph(data_path):
    args = load_config(os.path.join(data_path, 'config.txt'))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))
    h = hypergraph(args)
    h.load()
    return h

if __name__ == '__main__':
    args = parser.parse_args()
    if args.options is not None:
        args = load_config(args.options)
    if args.seed is not None:
        np.random.seed(args.seed)
    #    读数据集
    dataset = read_data_sets(args.data_path)
    # 特征维度
    args.dim_feature = [sum(dataset.train.nums_type)-n for n in dataset.train.nums_type]
    print("dataset.train.nums_type:",dataset.train.nums_type)
    # print("dataset.train.nums_type:",dataset.train.nums_type)
    print("args.dim_feature:",args.dim_feature)
    # tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
    # tf.ConfigProto一般用在创建session的时候，用来对session进行参数配置。
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.compat.v1.Session(config=config))
    # 构建超图
    h = hypergraph(args)

    begin = time.time()
    # 训练
    h.train(dataset)
    end = time.time()
    print("time, ", end-begin)
    # 保存模型
    h.save()
    # 保存embedding
    h.save_embeddings(dataset)
    # 清除模型缓存
    K.clear_session()
