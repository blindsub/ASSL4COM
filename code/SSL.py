#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：半监督性能预测 -> svm_ssl
@IDE    ：PyCharm
@Author ：zspp
@Date   ：2021/9/28 14:38
==================================================
"""
from sklearn.svm import SVC
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import accuracy_score
import math


class svm_ssl(object):
    def __init__(self, split_number_u, system, s_number):
        self.split_number_u = self.split_number_l = split_number_u  # deepperf所使用数据集大小，实际我们用的是一半
        self.system = system  # 系统名称
        self.s_number = 10  # 每次s伪造数据的大小

        self.test_number = 50
        self.flag = True  # 标记是否添加整体伪标签数据，true为否。
        self.SVM1 = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced', probability=True)
        self.SVM2 = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced', probability=True)
        # self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                '{}/svm_ssl_model.pickle'.format(system))
        # self.scalar_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                       '{}/svm_ssl_scalar.pickle'.format(system))

        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '{}/{}_test_data.csv'.format(system, system))
        self.data_L_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '{}/{}/{}_{}_data_L.csv'.format(system, s_number, system, self.split_number_u))

        self.data_U_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '{}/{}/{}_{}_data_U.csv'.format(system, s_number, system, self.split_number_u))

    def data_in(self, data):
        columns = list(data.columns)
        data = data.drop('label', 1)
        # print(data)
        return data

    # 训练SVM1
    def fit_svm1(self, data, model):
        """
        :param data: 一行为config1+config2+label
        :param model: 模型
        :return:
        """
        Y = data.iloc[:, -1].values.astype(int)
        if len(np.unique(Y)) == 1:
            data = pd.concat((data, data.iloc[-1:, :]), axis=0)
            Y = data.iloc[:, -1].values.astype(int)
            Y[-1] = abs(Y[-1] - 1)
        # 数据编码
        X = self.data_in(data)
        # X = data
        # 数据标准化
        # print(X)
        # print((Y))
        X = X.values
        scalar = StandardScaler()
        X = scalar.fit_transform(X)
        model = model.fit(X, Y)
        return model, scalar

    # 训练SVM2
    def fit_svm2(self, data, model):
        """
        :param data: 一行为config1+config2+label
        :param model: 模型
        :return:
        """
        Y = data.iloc[:, -1].values.astype(int)
        if len(np.unique(Y)) == 1:
            data = pd.concat((data, data.iloc[-1:, :]), axis=0)
            Y = data.iloc[:, -1].values.astype(int)
            Y[-1] = abs(Y[-1] - 1)
        # 数据编码
        X = self.data_in(data)
        # X = data

        # 数据标准化
        # print(X)
        # print((Y))
        X = X.values
        scalar = StandardScaler()
        X = scalar.fit_transform(X)
        model = model.fit(X, Y)
        return model, scalar

    def one_train(self, svm1, svm2, data_L, data_U, data_S):
        """
        迭代一次训练SVM1，SVM2
        :param svm1:模型
        :param svm2:模型
        :param data_L:dataframe
        :param data_U:dataframe
        :param data_S:dataframe
        :return:
        """
        # permutation = np.random.permutation(len(data_L))

        data_L1 = data_L.iloc[:int(len(data_L) / 2), :]
        data_L2 = data_L.iloc[int(len(data_L) / 2):, :]
        # 训练SVM1
        svm1_train_data = pd.concat((data_L1, data_S), axis=0)
        svm1_train_data = svm1_train_data.reset_index(drop=True)

        svm1, scalar1 = self.fit_svm1(svm1_train_data, svm1)

        # 训练SVM2
        svm2, scalar2 = self.fit_svm2(data_L2, svm2)

        # 伪标签S集合构造
        U_and_S_data = pd.concat((data_U, data_S), axis=0)
        U_and_S_data = U_and_S_data.reset_index(drop=True)

        test_data_process = self.data_in(U_and_S_data)
        # print(test_data_process.columns)
        # test_data_process =U_and_S_data

        test_data_process1 = scalar1.transform(test_data_process)
        test_data_process2 = scalar2.transform(test_data_process)

        predict1 = svm1.predict(test_data_process1)
        predict2 = svm2.predict(test_data_process2)

        # 置信度
        probility2 = svm1.predict_proba(test_data_process1)
        probility2 = np.array([max(i) for i in probility2])

        index_new_S = []
        i = 0
        for x, y in zip(list(predict1), list(predict2)):
            if x == y:
                index_new_S.append(i)
            i += 1

        data_S = U_and_S_data.iloc[index_new_S, :]  # 伪标签数据集构成
        probility2 = probility2[index_new_S]  # 置信度选择

        # print(data_S)
        predict_values = predict1[index_new_S]
        true_value = list(data_S.iloc[:, -1])

        data_S = data_S.iloc[:, :-1]
        data_S['label'] = np.array(predict_values)

        data_S = data_S.reset_index(drop=True)
        if len(data_S) > self.s_number and self.flag:
            # index = np.argpartition(probility2, len(probility2) - self.s_number)
            # index = index[-self.s_number:]
            # list_choose_s = index
            index = np.argsort(probility2)
            list_choose_s = index[-self.s_number:]
        else:
            list_choose_s = range(len(data_S))

        data_S = data_S.iloc[list_choose_s, :]
        data_S = data_S.reset_index(drop=True)

        dengyu = sum(predict_values[list_choose_s] == np.array(true_value)[list_choose_s])

        # print("伪标签的正确度为：{},伪标签大小为{}".format(dengyu / len(data_S), len(data_S)))

        # print(predict_values)
        # print(data_S)
        # data_S[]
        index_new_S = [index_new_S[i] for i in list_choose_s]  # 更改ssl数字要修改的部分
        data_U = U_and_S_data.drop(index=index_new_S)  # U中去掉加入到S中的
        data_U = data_U.reset_index(drop=True)
        data_L = data_L.reset_index(drop=True)

        return svm1, svm2, data_L, data_U, data_S, scalar1

    def train(self):
        # print("_________dataL读数据_______________")
        data_L = pd.read_csv(self.data_L_path)
        # data_L = data_L.sample(frac=split_number).reset_index(drop=True)

        # print("_________dataU读数据________________")

        data_U = pd.read_csv(self.data_U_path)
        # data_U = data_U.sample(frac=split_number).reset_index(drop=True)

        final_data_l = data_L.iloc[:, :]

        test_data = pd.read_csv(self.test_data_path)

        data_S = pd.DataFrame(columns=data_L.columns)
        epoch = 0
        # print("准备数据-----")
        # print("dataL初始大小为：{}".format(len(data_L)))
        # print("dataU初始大小为：{}".format(len(data_U)))
        # print("dataS初始大小为：{}".format(len(data_S)))
        while (not data_U.empty) and epoch < 10:
            # print("第{}epoch".format(epoch))

            self.SVM1, self.SVM2, data_L, data_U, data_S, scalar1 = self.one_train(self.SVM1, self.SVM2, data_L, data_U,
                                                                                   data_S)
            # print(data_U)
            epoch += 1
            # 训练中测试
            # print("dataL大小为：{}".format(len(data_L)))
            # print("dataU大小为：{}".format(len(data_U)))
            # print("dataS大小为：{}".format(len(data_S)))
            self.test(self.SVM1, test_data, scalar1)

        # 额外最后一次训练svm1，作为最终模型,此时dataL标签用全部数据

        svm1_train_data = pd.concat((data_L, data_S), axis=0)
        svm1_train_data = svm1_train_data.reset_index(drop=True)

        svm1, scalar = self.fit_svm1(svm1_train_data, self.SVM1)

        # 测试
        accuracy, rank_avg, rd_loss_max, rd_loss_min = self.test(svm1, test_data, scalar)
        print("测试集准确率为：{}".format(accuracy))
        print("rank损失(越小越好)为：{}".format(rank_avg))
        print("rd损失(越小越好,性能值越大越好的系统)为：{}".format(rd_loss_max))
        print("rd损失(越小越好,性能值越小越好的系统)为：{}".format(rd_loss_min))
        # return svm1, test_data, scalar
        return accuracy, rank_avg, rd_loss_max, rd_loss_min

    def rank_test(self, predict, true_y, test_number):
        """
        bad learner 中的rank损失以及rd损失
        Args:
            predict:   对比预测结果
            test_number:  测试集大小，比如50条
        Returns:
        """
        predict_rank = {}  # 记录每个值比他大以及比他小的个数1:[23,27]
        true_rank = {}
        for i in range(test_number):
            predict_rank[i] = [0, 0]
            true_rank[i] = [0, 0]

        number_index = 0
        for i in range(test_number):
            for j in range(i + 1, test_number):
                if predict[number_index] == 1:
                    predict_rank[i][1] += 1
                    predict_rank[j][0] += 1
                else:
                    predict_rank[i][0] += 1
                    predict_rank[j][1] += 1

                if true_y[number_index] == 1:
                    true_rank[i][1] += 1
                    true_rank[j][0] += 1
                else:
                    true_rank[i][0] += 1
                    true_rank[j][1] += 1
                number_index += 1

        # rank准确率计算
        rank_avg = 0
        rd_loss_max = 0
        rd_loss_min = 0

        for i in range(len(predict_rank)):
            rank_avg += abs(predict_rank[i][0] - true_rank[i][0])
        rank_avg = rank_avg / len(predict_rank)
        # print("deepperf在test_data上的rank损失(越小越好)为：{}".format(rank_avg))

        for i in range(len(predict_rank)):
            if true_rank[i][1] == 0:
                rd_loss_max = abs(predict_rank[i][0] - true_rank[i][0])
                # print("deepperf在test_data上的rd损失(越小越好,性能值越大越好的系统)为：{},i为{}".format(rd_loss_max, i))
                break

        for i in range(len(predict_rank)):
            if true_rank[i][0] == 0:
                rd_loss_min = abs(predict_rank[i][0] - true_rank[i][0])
                # print("deepperf在test_data上的rd损失(越小越好,性能值越小越好的系统)为：{},i为{}".format(rd_loss_min, i))
                break

        return rank_avg, rd_loss_max, rd_loss_min

    def test(self, model, testdata, scalar):
        y = testdata.iloc[:, -1].values.astype(int)

        testdataset = self.data_in(testdata)
        testdataset = testdataset.reset_index(drop=True)

        X = scalar.transform(testdataset.values)
        y_pred = model.predict(X)
        # print(classification_report(y, y_pred))
        # print("测试数据准确率为：", accuracy_score(y, y_pred))
        rank_avg, rd_loss_max, rd_loss_min = self.rank_test(y_pred, y, self.test_number)
        # print("rank损失(越小越好)为：{}".format(rank_avg))
        # print("rd损失(越小越好,性能值越大越好的系统)为：{}".format(rd_loss_max))
        # print("rd损失(越小越好,性能值越小越好的系统)为：{}".format(rd_loss_min))
        return accuracy_score(y, y_pred), rank_avg, rd_loss_max, rd_loss_min


def system_samplesize(sys_name):

    N_train_all = np.multiply(6, [1, 2, 3])

    return N_train_all


def out_dict_info():
    """
        dict_data = {2:100: {
                           'mysql': {10: [45, 5], 20: [-8, 8], 30: [-5, 6], 40: [-10, 5], 50: [-6, 4]},
                           'redis': {9: [36, 5], 18: [136, 8], 27: [-19, 7], 36: [253, 5], 45: [-25, 5]},
                           'sqlite': {22: [-16, 8], 44: [-20, 5], 66: [-8, 3], 88: [-35, 3], 110: [-11, 2]},
                           'tomcat': {12: [66, 6], 24: [-5, 7], 36: [253, 5], 48: [-2, 4], 60: [-26, 4]},
                           'x264': {9: [36, 5], 18: [136, 8], 27: [-19, 7], 36: [253, 5], 45: [-25, 5]},

                           'hadoopsort': {9: [36, 5], 18: [136, 8], 27: [-19, 7], 36: [253, 5], 45: [-25, 5]},
                           'hadoopterasort': {9: [36, 5], 18: [136, 8], 27: [-19, 7], 36: [253, 5], 45: [-25, 5]},
                           'hadoopwordcount': {9: [36, 5], 18: [136, 8], 27: [-19, 7], 36: [253, 5], 45: [-25, 5]},
                           'sparksort': {13: [-15, 10], 26: [-12, 7], 39: [-10, 5], 52: [-10, 4], 65: [-8, 3]},
                           'sparkterasort': {13: [-15, 10], 26: [-12, 7], 39: [-10, 5], 52: [-10, 4], 65: [-8, 3]},
                           },
        :return:
        """
    # 对于1N不充分的，选择全部添加构造数据集
    systerm_list = ['hadoopsort', 'hadoopterasort', 'hadoopwordcount', 'mysql',
                    'redis', 'sparksort', 'sparkterasort', 'sparkwordcount', 'sqlite', 'tomcat', 'x264']

    s_number_list = [2, 3, 4]

    class multidict(dict):
        def __getitem__(self, item):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    dict_info = multidict()

    for s_number in s_number_list:
        for system in systerm_list:
            number_list = system_samplesize(system)
            for number in number_list:

                if number % s_number != 0:
                    half = math.floor(number / s_number)
                    Q_number = (number - math.floor(number / s_number)) * 20  # 计算专家次数
                else:
                    half = number / s_number
                    Q_number = math.ceil(number / s_number) * (s_number - 1) * 20  # 计算专家次数

                N = (half * (half - 1) // 2) + Q_number
                first = math.ceil(math.sqrt(N * 2))
                if first > number:
                    add_ = number - half
                    sub_ = (number) * (number - 1) // 2
                else:
                    if first * (first - 1) // 2 < N:
                        first += 1

                        add_ = first - half
                        if first * (first - 1) // 2 == N:
                            sub_ = first * (first - 1) // 2
                        else:
                            sub_ = first * (first - 1) // 2 - N
                            sub_ = - sub_
                    elif first * (first - 1) // 2 == N:
                        add_ = first - half
                        sub_ = first * (first - 1) // 2
                    else:
                        add_ = first - half
                        if first * (first - 1) // 2 == N:
                            sub_ = first * (first - 1) // 2
                        else:
                            sub_ = first * (first - 1) // 2 - N
                            sub_ = - sub_
                dict_info[s_number][system][Q_number][number] = [sub_, add_]
    return dict_info

if __name__ == '__main__':

    dict_data = out_dict_info()
    systerm_list = ['hadoopsort', 'hadoopterasort', 'hadoopwordcount', 'sparksort', 'sparkterasort', 'sparkwordcount',
                    'mysql', 'redis',
                    'x264', 'tomcat', 'sqlite']
    s_number_list = [2, 3, 4]

    columns = ['Systerm', 'Split_NUMBER', 'NUMBER', 'accuracy', 'rank_avg',
               'rd_loss_max', 'rd_loss_min']
    out = pd.DataFrame(columns=columns)
    out.to_csv('svm_ssl.csv', index=False)

    for system in systerm_list:
        for s_number in s_number_list:
            number_list = system_samplesize(system)
            for number in number_list:
                if number==6:
                    continue
                print("---{}---{}----{}-".format(system, s_number, number))
                accuracy, rank_avg, rd_loss_max, rd_loss_min = svm_ssl(number, system, s_number).train()
                row = [
                    [system, s_number, number, accuracy, rank_avg, rd_loss_max,
                     rd_loss_min]]
                out = pd.DataFrame(row)
                out.to_csv('svm_ssl.csv', index=False, mode='a+', header=None)
                print("----------------------")
