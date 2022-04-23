#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：半监督性能预测 -> train
@IDE    ：PyCharm
@Author ：zspp
@Date   ：2021/9/23 11:06
==================================================
"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
import math


class svm_our(object):
    def __init__(self, split_number_u, system, q_number, SPLIT_NUMBER):
        self.split_number_u = self.split_number_l = split_number_u  # deepperf所使用数据集大小，实际我们用的是一半
        self.system = system  # 系统名称

        self.SVM = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced')
        self.test_number = 50
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}/svm_model.pickle'.format(system))
        self.scalar_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '{}/svm_scalar.pickle'.format(system))

        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '{}/{}_test_data.csv'.format(system, system))

        self.svm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '{}/{}/{}/{}_svm_{}_data_U.csv'.format(system, SPLIT_NUMBER, q_number, system,
                                                                            self.split_number_u))

    def data_in(self, data):
        """
        # 数据编码
        :param data:
        :return: dataframe
        """
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
        # 数据编码
        X = self.data_in(data)

        SCALAR = StandardScaler()
        X = SCALAR.fit_transform(X)

        # print("0的个数：", np.sum(Y == 0))
        # print("1的个数：", np.sum(Y == 1))
        scores = cross_val_score(model, X, Y, cv=3, scoring='accuracy')
        # print("交叉验证scores:", scores)
        model = model.fit(X, Y)

        y_pred = model.predict(X)
        # print("训练集准确率——————————————————")
        # print(classification_report(Y, y_pred))
        # print("准确率为：", accuracy_score(Y, y_pred))

        return model, SCALAR

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

    def train(self):
        # print("_________读训练数据_______________")
        data_L = pd.read_csv(self.svm_path)

        # print("dataL初始大小为：{}".format(len(data_L)))
        svm1, SCALAR = self.fit_svm1(data_L, self.SVM)
        # 测试
        accuracy, rank_avg, rd_loss_max, rd_loss_min = self.test(svm1, SCALAR)
        print("测试集准确率为：{}".format(accuracy))
        print("rank损失(越小越好)为：{}".format(rank_avg))
        print("rd损失(越小越好,性能值越大越好的系统)为：{}".format(rd_loss_max))
        print("rd损失(越小越好,性能值越小越好的系统)为：{}".format(rd_loss_min))
        return accuracy, rank_avg, rd_loss_max, rd_loss_min
        # return svm1, SCALAR

    def test(self, model, SCALAR):
        # print("_________测试数据____________")
        testdataset = pd.read_csv(self.test_data_path)

        y = testdataset.iloc[:, -1].values.astype(int)

        testdataset = self.data_in(testdataset)
        # testdataset = substract_(testdataset)
        X = testdataset.reset_index(drop=True)

        X = SCALAR.transform(X)
        y_pred = model.predict(X)

        # print(classification_report(y, y_pred))
        # print("测试集准确率为：{}".format(accuracy_score(y, y_pred)))
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

    columns = ['Systerm', 'Split_NUMER', 'expert_num', 'NUMBER', 'accuracy', 'rank_avg', 'rd_loss_max', 'rd_loss_min']
    out = pd.DataFrame(columns=columns)
    out.to_csv('svm.csv', index=False)

    for system in systerm_list:
        for s_number in s_number_list:
                number_list = system_samplesize(system)
                for number in number_list:

                    if number==6:
                        continue

                    q_number = (number - math.floor(number / s_number)) * 20  # 计算专家次数

                    print("---{}---{}---{}--{}-".format(system, s_number, q_number, number))
                    accuracy, rank_avg, rd_loss_max, rd_loss_min = svm_our(number, system, q_number, s_number).train()
                    row = [[system, s_number, accuracy, q_number, number, rank_avg, rd_loss_max, rd_loss_min]]
                    out = pd.DataFrame(row)
                    out.to_csv('svm.csv', index=False, mode='a+', header=None)
                    print("----------------------")
