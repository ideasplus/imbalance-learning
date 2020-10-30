import numpy as np
from collections import Counter

from sklearn import tree


class EasyEnsemble(object):

    def __init__(self, T, rounds):
        """
        初始化
        :param T: the number of subsets T to sample from N
        :param rounds: the number of iterations to train an AdaBoost ensemble H[i]
        """
        self.T = T
        self.rounds = rounds

        self.minority_target = None
        self.majority_target = None
        # 最终的强分类器
        self.ensemble = {'trees': [], 'alpha': [], 'threshold': []}

    @staticmethod
    def boost_data(data, label, weight):
        """
        To sample a subset according to each example's weight
        :param data: n-by-d training set
        :param label: n-by-1 training target
        :param weight: weight for sample
        :return: boostset: sampled data set
                 boosttarget: labels for boostset
        """
        size = len(data)
        cumulative_sum = np.cumsum(weight)
        # 产生[0,1]之间的随机数
        select = np.random.rand(len(label))
        select_idx = np.zeros(len(label), dtype=np.int)
        # 根据权重采样
        for i in range(size):
            select_idx[i] = np.min(np.flatnonzero(cumulative_sum >= select[i]))

        boost_set = data[select_idx]
        boost_target = label[select_idx]
        return boost_set, boost_target

    def adaboost(self, data, label):
        """
        train an Adaboost classifier
        :param data: n-by-d training set
        :param label: n-by-1 training target
        :return: AdaBoost classifier, a structure variable
        """
        boost = {'trees': [], 'alpha': [0] * self.rounds, 'threshold': []}

        # initialize weight
        weight = np.zeros(len(data))
        # assign weight for minority
        weight[label == self.minority_target] = 1 / np.sum(label == self.minority_target)
        # assign weight for majority
        weight[label == self.majority_target] = 1 / np.sum(label == self.majority_target)
        # average
        weight = weight / np.sum(weight)

        for i in range(self.rounds):
            # 初始化分类器
            classifier = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
            # 重采样样本
            boost_set, boost_target = self.boost_data(data, label, weight)
            # train base classifier
            clf = classifier.fit(boost_set, boost_target)
            boost['trees'].append(clf)
            predict = clf.predict(data)
            # calculate error rate
            error = np.sum(weight * (predict != label))
            # print(error)
            beta = (1 - error) / error
            boost['alpha'][i] = 0.5 * np.log(beta)
            # update weight
            weight = weight * np.exp(-boost['alpha'][i] * (predict - 0.5) * (label - 0.5) * 4)
            weight = weight / np.sum(weight)
        boost['threshold'] = np.sum(boost['alpha']) / 2

        return boost

    def evaluateValue(self, X):
        """
        计算测试样本的输出值
        :param X: test data
        :return:
        """
        value = np.zeros(len(X))
        for i in range(len(self.ensemble['trees'])):
            value = value + self.ensemble['alpha'][i] * self.ensemble['trees'][i].predict(X)

        return value

    def CalculatePositives(self, test_data, test_label):
        """
        To calculate (fpr,tpr)
        :param test_data: 测试样本
        :param test_label: 测试标签
        :return: rates: (fpr,tpr) vector with fpr in ascending order
        """
        # determine the minority class label
        stats_c_ = Counter(test_label)
        majority_target = max(stats_c_, key=stats_c_.get)
        # convert
        if majority_target != 0:
            test_label[test_label == majority_target] = 0

        size = len(test_label)
        values = self.evaluateValue(test_data)
        vi = np.c_[values, test_label]
        vi = vi[vi[:, 0].argsort(),]

        fp = np.zeros(size + 1)
        tp = np.zeros(size + 1)

        tpc = len(np.flatnonzero(test_label == self.minority_target))
        fpc = size - tpc
        prev = -1
        index = 0
        for i in range(size):
            if vi[i, 0] != prev:
                prev = vi[i, 0]
                tp[index] = tpc
                fp[index] = fpc
                index = index + 1
            if vi[i, 1] == self.minority_target:
                tpc = tpc - 1
            else:
                fpc = fpc - 1

        tp[index] = 0
        fp[index] = 0
        tp = tp[0:index]
        fp = fp[0:index]

        rates = np.c_[fp, tp]
        rates = np.flipud(rates)
        rates[:, 0] = rates[:, 0] / (len(test_label) - np.sum(test_label))
        rates[:, 1] = rates[:, 1] / np.sum(test_label)

        return rates

    def CalculateAUC(self, test_data, test_label):
        """
        To calculate AUC values
        :param test_data: 测试数据
        :param test_label: 测试标签
        :return: AUC value
        """
        # determine the minority class label
        stats_c_ = Counter(test_label)
        majority_target = max(stats_c_, key=stats_c_.get)
        # convert
        if majority_target != 0:
            test_label[test_label == majority_target] = 0

        rates = self.CalculatePositives(test_data, test_label)
        AUC = 0
        for i in range(len(rates) - 1):
            AUC = AUC + (rates[i + 1, 0] - rates[i, 0]) * (rates[i + 1, 1] + rates[i + 1, 1]) / 2
        return AUC

    @staticmethod
    def ImbalanceEvaluate(true_labels, pred_labels, pclass, nclass):
        """
        计算评价指标
        :param true_labels: 分类标签
        :param pred_labels: 预测标签
        :param pclass: 正类标号
        :param nclass: 负类标号
        :return: F1 and G-mean
        """
        # determine the minority class label
        stats_c_ = Counter(true_labels)
        majority_target = max(stats_c_, key=stats_c_.get)
        # convert
        if majority_target != 0:
            true_labels[true_labels == majority_target] = 0

        TP = np.sum(np.logical_and(pred_labels == pclass, true_labels == pclass))
        TN = np.sum(np.logical_and(pred_labels == nclass, true_labels == nclass))
        FP = np.sum(np.logical_and(pred_labels == pclass, true_labels == nclass))
        FN = np.sum(np.logical_and(pred_labels == nclass, true_labels == pclass))

        if TP == 0:
            precision = 0
            recall = 0
            f1 = 0
            g_mean = 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)

            Acc_pos = recall
            Acc_neg = TN / (TN + FP)
            g_mean = np.sqrt(Acc_pos * Acc_neg)

        return precision, recall, f1, g_mean

    def fit(self, X, y):
        # determine the minority class label
        stats_c_ = Counter(y)
        self.minority_target = min(stats_c_, key=stats_c_.get)
        self.majority_target = max(stats_c_, key=stats_c_.get)
        # convert
        if self.majority_target != 0:
            y[y == self.majority_target] = 0
            self.majority_target = 0

        # split X to positive and negative data
        pos_data = X[y == self.minority_target]
        neg_data = X[y == self.majority_target]
        pos_size = len(pos_data)
        neg_size = len(neg_data)

        # randomize negative data
        shuffle_indices = np.random.permutation(np.arange(neg_size))
        neg_data = neg_data[shuffle_indices]

        bagging = {}
        for node in range(self.T):
            # Randomly sample a subset from negative data
            nset = neg_data[0:pos_size]
            cur_trainset = np.vstack([pos_data, nset])
            cur_target = np.array([self.minority_target] * pos_size + [self.majority_target] * pos_size)
            # node classifier
            hypothesi = self.adaboost(cur_trainset, cur_target)
            bagging[node] = hypothesi
            # randomize negative data
            shuffle_indices = np.random.permutation(np.arange(neg_size))
            neg_data = neg_data[shuffle_indices]

        # combine all weak learners to form the final ensemble
        for i in range(len(bagging)):
            self.ensemble['trees'].extend(bagging[i]['trees'])
            self.ensemble['alpha'].extend(bagging[i]['alpha'])
        self.ensemble['threshold'] = np.sum(self.ensemble['alpha']) / 2

    def predict(self, X):
        """
        对测试集进行预测
        :param X: test data
        :return: result
        """
        value = self.evaluateValue(X)
        re = value >= self.ensemble['threshold']

        return re
