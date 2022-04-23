#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
=================================================
@Project -> File   ：半监督性能预测 -> svm+m_al+k_ssl
@IDE    ：PyCharm
@Author ：zspp
@Date   ：2021/10/15 15:40
==================================================
"""
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
import pandas as pd
import pickle
import numpy as np
import os
import copy
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import accuracy_score
import random
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import argparse
import math
from sklearn.cluster import KMeans


class svm_al_ssl_balance(object):
    def __init__(self, split_number_u, system, q_number, s_number):
        self.Q_number = 10  # 每次Q数据的大小
        self.RSTRICT_NUMBER = q_number  # 专家总的询问次数
        self.test_number = 50
        self.SVM1 = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced', probability=True)
        self.SVM2 = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced', probability=True)
        self.al_model_list = []
        self.cluster_number = 20  # 聚类数目

        self.split_number_u = self.split_number_l = split_number_u  # deepperf所使用数据集大小，实际我们用的是一半
        self.system = system
        self.s_number = s_number

        # self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}/svm_al_model.pickle'.format(system))
        # self.scalar_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                  '{}/svm_al_scalar.pickle'.format(system))
        self.data_L_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        '{}/{}/{}_{}_data_L.csv'.format(system, s_number, system, self.split_number_u))

        self.data_U_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        '{}/{}/{}_{}_data_U.csv'.format(system, s_number, system, self.split_number_u))

        self.test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           '{}/{}_test_data.csv'.format(system, system))

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
            if np.unique(Y)[0] == 1:
                Y[-1] = 0
            else:
                Y[-1] = 1
        # 数据编码
        X = self.data_in(data)
        # 数据标准化
        # print(X)
        # print((Y))
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
            if np.unique(Y)[0] == 1:
                Y[-1] = 0
            else:
                Y[-1] = 1
        # 数据编码
        X = self.data_in(data)
        # 数据标准化
        # print(X)
        # print((Y))

        scalar = StandardScaler()
        X = scalar.fit_transform(X)
        model = model.fit(X, Y)
        return model, scalar

    def Euclidean(self, point1, clusters):
        """
        计算欧式距离
        :param point1:array
        :param clusters:array
        :return:
        """
        dist = filter(lambda x: np.linalg.norm(point1 - x), clusters)
        return dist

    def train_cluster(self, data_u):
        """

        :param data_u:array
        :return:
        """
        cluster_model = KMeans(n_clusters=self.cluster_number).fit(data_u)
        clusters = cluster_model.cluster_centers_
        probility = filter(lambda x: self.Euclidean(x, clusters), data_u)
        probility = np.array([min(i) for i in probility])
        if len(probility) <= self.cluster_number:
            index = range(len(probility))
        else:
            index = np.argpartition(probility, self.cluster_number)

            index = index[:self.cluster_number]
        return index

    def one_train_al(self, svm1, data_L, data_U, expert_exact_percent):
        """
        迭代一次训练SVM1
        :param svm1:模型
        :param data_L:dataframe
        :param data_U:dataframe
        :return:
        """
        svm1, scalar1 = self.fit_svm1(data_L, svm1)
        data_U_process = self.data_in(data_U)

        data_U_process = scalar1.transform(data_U_process)

        # 聚类
        index_20 = self.train_cluster(data_U_process)
        data_cluster_20 = data_U_process[index_20]

        probility = svm1.predict_proba(data_cluster_20)
        probility = np.array([max(i) for i in probility])
        if len(probility) <= self.Q_number:
            index = range(len(probility))
        else:
            index = np.argpartition(probility, self.Q_number)

            index = index[:self.Q_number]

        # 人工标注
        Q_dataset = data_U.iloc[index_20[index], :]
        Q_dataset = Q_dataset.reset_index(drop=True)
        # print(Q_dataset)
        true_y_0 = np.sum(Q_dataset.iloc[:, -1] == 0)
        true_y_1 = np.sum(Q_dataset.iloc[:, -1] == 1)

        true_number = len(Q_dataset)
        for i in range(len(Q_dataset)):
            random_number = random.uniform(0, 1)
            if random_number > expert_exact_percent:
                Q_dataset.iloc[i, -1] = abs(1 - Q_dataset.iloc[i, -1])
                true_number -= 1

        # 选择的10个数据的当前模型给的标签
        predict_label = svm1.predict(Q_dataset.iloc[:, :-1].values)
        # print(predict_label)
        predict_y_0 = np.sum(predict_label == 0)
        predict_y_1 = np.sum(predict_label == 1)

        cha = np.sum(predict_label == Q_dataset.iloc[:, -1])

        # print('专家本身准确率\\专家实际准确率\\每轮专家标签类别\\真实标签类别\\模型给的标签类别\\模型和真实类别差：*{},{},[{},{}],[{},{}],[{},{}],{}*'.format(
        #     expert_exact_percent,
        #     true_number / len(Q_dataset),
        #     np.sum(Q_dataset.iloc[:, -1] == 0),
        #     np.sum(Q_dataset.iloc[:, -1] == 1),
        #     true_y_0,
        #     true_y_1, predict_y_0, predict_y_1, cha
        # ))
        # print()

        data_L = pd.concat((data_L, Q_dataset), axis=0)  # 原始样本集扩充真实样本集
        data_U = data_U.drop(index=index_20[index])  # U中去掉人工标注的
        data_U = data_U.reset_index(drop=True)
        data_L = data_L.reset_index(drop=True)

        return svm1, data_L, data_U, scalar1

    def one_train(self, svm1, svm2, data_L, data_U, Flag, scalar1, ssl_number, expert_exact_percent):
        """
        迭代一次训练SVM1，SVM2
        :param svm1:模型
        :param svm2:模型
        :param data_L:dataframe
        :param data_U:dataframe
        :return:
        """
        # al训练
        if Flag:
            svm1, data_L, data_U, scalar1 = self.one_train_al(svm1, data_L, data_U, expert_exact_percent)
            self.al_model_list.append(copy.deepcopy(svm1))
        else:

            # 标签变化度
            data_U_process = self.data_in(data_U)

            # data_U_process = choose.transform(data_U_process)
            data_U_process = scalar1.transform(data_U_process)

            predict_array = []

            for model in self.al_model_list:
                predict_values = model.predict(data_U_process)
                predict_array.append(predict_values)
                probility = model.predict_proba(data_U_process)
                probility = np.max(probility, axis=1)

            predict_array_sum = np.sum(np.array(predict_array), axis=0)
            # print(predict_array_sum)

            # 标签变化度选择变化为0的
            index = []
            for i, j in enumerate(predict_array_sum.ravel()):
                if j == 0 or j == M:
                    index.append(i)

            # 标签为1的选择
            index_one = [i for i in index if predict_array[-1][i] == 1]
            # 标签为0的选择
            index_zero = [i for i in index if predict_array[-1][i] == 0]
            # print('标签未变化，且预测标签为0的有{}个'.format(len(index_zero)))
            # print('标签未变化，且预测标签为1的有{}个'.format(len(index_one)))

            if len(index_one) < int(ssl_number / 2):
                actual_index1 = index_one
            else:
                # 伪标签1选取置信度最大的
                rank1 = np.argsort(probility[index_one])
                mid_length = int(len(rank1) / 3)

                index_new1 = rank1[-int(ssl_number / 2):]

                actual_index1 = [index_one[i] for i in index_new1]

            # 伪标签0选取置信度最大的
            if len(index_zero) < int(ssl_number / 2):
                actual_index0 = index_zero
            else:
                rank0 = np.argsort(probility[index_zero])
                mid_length = int(len(rank0) / 3)

                index_new0 = rank0[-int(ssl_number / 2):]

                actual_index0 = [index_zero[i] for i in index_new0]

            actual_index = actual_index0 + actual_index1

            # data_s构造
            new_data_S = data_U.iloc[actual_index, :]
            predict_value = [predict_array[-1][i] for i in actual_index]
            true_value = list(new_data_S.iloc[:, -1])

            new_data_S = new_data_S.iloc[:, :-1]
            new_data_S['label'] = np.array(predict_value)

            data_U = data_U.drop(index=actual_index)
            data_L = pd.concat((data_L, new_data_S), axis=0)
            data_L = data_L.reset_index(drop=True)
            data_U = data_U.reset_index(drop=True)

            # dengyu = sum(np.array(predict_value) == np.array(true_value))
            # print("预测伪标签中0的个数：", np.sum(np.array(predict_value) == 0), "真实伪标签中0的个数：", np.sum(np.array(true_value) == 0))
            # print("预测伪标签中1的个数：", np.sum(np.array(predict_value) == 1), "真实伪标签中1的个数：", np.sum(np.array(true_value) == 1))
            # print("伪标签的正确度为：{},伪标签大小为{}".format(dengyu / len(new_data_S), len(new_data_S)))

        return svm1, svm2, data_L, data_U, scalar1

    def train(self, ssl_number, all_k, M, expert_exact_percent):
        # print("_________dataL读数据_______________")
        data_L = pd.read_csv(self.data_L_path)
        # print(data_L_path)
        data_U = pd.read_csv(self.data_U_path)
        # data_U = data_U.sample(frac=split_number).reset_index(drop=True)

        if len(data_U) < self.RSTRICT_NUMBER + all_k * ssl_number:  # 判断是否有足够的伪标签提供
            return -1, -1, -1, -1
        test_data = pd.read_csv(self.test_data_path)

        data_S = pd.DataFrame(columns=data_L.columns)

        epoch = 0
        restrict_number = 0
        # print("准备数据-----")
        # print("dataL初始大小为：{}".format(len(data_L)))
        # print("dataU初始大小为：{}".format(len(data_U)))
        # print("dataS初始大小为：{}".format(len(data_S)))
        scalar1 = None
        choose = None
        m = 0
        start_k = 0
        while (restrict_number < self.RSTRICT_NUMBER and not data_U.empty) or start_k < all_k or m < M:
            # print("第{}epoch".format(epoch))
            epoch += 1

            if epoch % (M + 1) == 0 and epoch != 0:
                Flag = False
                start_k += 1
            else:
                Flag = True
                m += 1
                restrict_number += self.Q_number
            self.SVM1, self.SVM2, data_L, data_U, scalar1 = self.one_train(self.SVM1, self.SVM2, data_L, data_U, Flag,
                                                                           scalar1, ssl_number,
                                                                           expert_exact_percent)

            # 训练中测试
            # print("第{}epoch".format(epoch))
            # print("dataL大小为：{}".format(len(data_L)))
            # print("dataU大小为：{}".format(len(data_U)))
            # print("dataS大小为：{}".format(len(data_S)))
            # self.test(self.SVM1, test_data, scalar1)

        # 额外最后一次训练svm1，作为最终模型
        svm1_train_data = pd.concat((data_L, data_S), axis=0)
        svm1_train_data = svm1_train_data.reset_index(drop=True)
        self.SVM1, scalar = self.fit_svm1(svm1_train_data, self.SVM1)

        # 测试
        ac, rk, rdmin, rdmax = self.test(self.SVM1, test_data, scalar)
        print(
            "M={},K={},split={},S_NUMBER={},expert={},准确率\\rank loss\\rd_min\\rd_max为：".format(M, all_k, self.s_number,
                                                                                               ssl_number,
                                                                                               expert_exact_percent),
            ac, rk,
            rdmin,
            rdmax)
        # return svm1, test_data, scalar
        return round(ac, 8), rk, rdmin, rdmax

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
        # print(true_rank)
        # print(predict_rank)
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
        # print("_________测试数据____________")

        y = testdata.iloc[:, -1].values.astype(int)
        testdataset = self.data_in(testdata)
        testdataset = testdataset.reset_index(drop=True)
        X = scalar.transform(testdataset)
        y_pred = model.predict(X)
        # print(classification_report(y, y_pred))
        # print(y_pred[:100])
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


def out_dict_mk():
    """
    计算m和k
    """
    dict_mk = {
        60: [[2, 3, 6], [3, 2, 1]],
        280: [[2, 3, 4, 7, 9, 14, 16, 18, 20, 24, 28], [14, 9, 7, 4, 3, 2, 1, 1, 1, 1, 1]],
        260:[[2, 3, 4, 5, 9, 13, 16, 18, 20, 24, 26], [13, 8, 6, 5, 2, 2, 1, 1, 1, 1, 1]],
        240: [[2, 3, 4, 6, 8, 12, 14, 16, 18, 24], [12, 8, 6, 4, 3, 2, 1, 1, 1, 1]],
        180: [[2, 3, 5, 6, 9, 12, 14, 16, 18], [9, 6, 3, 3, 2, 1, 1, 1, 1]],
        160: [[2, 4, 5, 7, 8, 12, 14, 16], [8, 4, 3, 2, 2, 1, 1, 1]],
        120: [[2, 3, 4, 5, 7, 10, 12], [6, 4, 3, 2, 1, 1, 1]],
        100: [[2, 3, 4, 5, 7, 10], [5, 3, 2, 2, 1, 1]],
        80: [[2, 4, 8], [4, 2, 1]],
        50: [[1, 5], [5, 1]],
    }
    return dict_mk


if __name__ == '__main__':

    dict_data = out_dict_info()
    systerm_list = ['hadoopsort', 'hadoopterasort', 'hadoopwordcount', 'sparksort', 'sparkterasort', 'sparkwordcount',
                    'mysql', 'redis',
                    'x264', 'tomcat', 'sqlite']
    s_number_list = [2, 3, 4]

    expert_exact_percent_list = [1, 0.9, 0.8, 0.7]
    ssl_number_list = [10, 100, 1000, 3000, 5000]

    columns = ['expert_exact_percent', 'Systerm', 'Split_NUMBER', 'expert_num', 'NUMBER', 'accuracy', 'rank_avg',
               'rd_loss_max', 'rd_loss_min', 'M', 'k', 'SSL_max']
    out = pd.DataFrame(columns=columns)
    out.to_csv('svm_m_al_k_ssl_(对比).csv', index=False)

    dict_mk = out_dict_mk()
    for expert_exact_percent in expert_exact_percent_list:
        for system in systerm_list:
            for s_number in s_number_list:
                    number_list = system_samplesize(system)
                    for number in number_list:

                        if number == 6:
                            continue

                        q_number = (number - math.floor(number / s_number)) * 20  # 计算专家次数

                        OUT_LIST = [0, 0, 0, 0, 0, 0, 0]

                        print("--专家准确率:{}- {}---{}---{}--{}-".format(expert_exact_percent, system, s_number,
                                                                     q_number, number
                                                                     ))
                        for i in range(len(dict_mk[q_number][0])):
                            M = dict_mk[q_number][0][i]
                            all_k = dict_mk[q_number][1][i]
                            # 添加伪标签次数
                            for ssl_number in ssl_number_list:
                                # 训练模型
                                accuracy_s, rk, rdmin, rdmax = svm_al_ssl_balance(number, system, q_number,
                                                                                  s_number).train(ssl_number, all_k, M,
                                                                                                  expert_exact_percent)
                                # if accuracy_s > OUT_LIST[0]:
                                    # OUT_LIST = [round(accuracy_s, 8), rk, rdmin, rdmax, M, all_k, ssl_number]
                                OUT_LIST = [round(accuracy_s, 8), rk, rdmin, rdmax, M, all_k, ssl_number]

                                row = [
                                    [expert_exact_percent, system, s_number, q_number, number, OUT_LIST[0], OUT_LIST[1],
                                     OUT_LIST[2],
                                     OUT_LIST[3], OUT_LIST[4], OUT_LIST[5], OUT_LIST[6]]]
                                out = pd.DataFrame(row)
                                out.to_csv('svm_m_al_k_ssl_(对比).csv', index=False, mode='a+', header=None)
                                print('--------------------------------------------')
        break