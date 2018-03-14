#!/usr/bin/env python3
# coding=utf-8
"""
The script helps guide the users to quickly understand how to use
libact by going through a simple active learning task with clear
descriptions.
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler

#The main part that uses libact is in the run function:
def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []
    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()  #returns the index of the sample that the active learning algorithm wants to query.
        X, _ = zip(*trn_ds.data)
        print X
        print _
        lb = lbr.label(X[ask_id])#lbr acts as the oracle#returns the label of the given sample answered by oracle.
        trn_ds.update(ask_id, lb)#updates the unlabeled sample with queried label.
#A common way of evaluating the performance of active learning algorithm is to plot the learning curve. 
        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))#in-sample error
        E_out = np.append(E_out, 1 - model.score(tst_ds))#out-sample error

    return E_in, E_out

#First, the data are splitted into training and testing set:
def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)
    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds
    #trn_ds与fullly相比完全相同，不过fully是全标记，trn_ds只标记了labeled个，格式是[array(标记元素),label]
    #y_train是fully_labeled的label，list
    #tst_ds是测试集，1-fully

def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    dataset_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'diabetes1.txt')
    test_size = 0.33    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 2      # number of samples that are initially labeled

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)

    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b', label='qs Ein')
    plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='qs Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
