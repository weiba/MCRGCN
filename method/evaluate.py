import scipy.io as sio
import sys
import numpy as np
from sklearn import metrics,svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, auc, precision_recall_curve, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
import torch
from sklearn.model_selection import KFold

def evaluate(embeds,labels,indexes):

    s_p = 0
    s_r = 0
    s_f1score = 0
    s_ACC = 0
    s_AUC = 0
    s_PR_AUC = 0
    s_ARS = 0
    s_DBI = 0
    s_SS = 0
    s_MCC = 0
    mask = torch.randperm(len(indexes))
    path = sys.path[1]
    f_name = '/results_save_gbm/'
    for n in range(10):
        print(n)
        p = np.zeros(5)
        r = np.zeros(5)
        f1score = np.zeros(5)
        ACC = np.zeros(5)
        AUC = np.zeros(5)
        PR_AUC = np.zeros(5)
        ARS = np.zeros(5)
        DBI = np.zeros(5)
        SS = np.zeros(5)
        MCC = np.zeros(5)
        k=0
        kfold = KFold(n_splits=5, shuffle=True, random_state=n * n + 1)
        for train_index, test_index in kfold.split(mask):
            x_train = embeds[train_index]
            targets_train = labels[train_index]
            x_test = embeds[test_index]
            targets_test = labels[test_index]

            x_train=x_train.cuda().data.cpu().numpy()
            x_test =  x_test.cuda().data.cpu().numpy()
            targets_train = targets_train.cuda().data.cpu().numpy()
            targets_test=targets_test.cuda().data.cpu().numpy()

            classifier = MLPClassifier(activation='tanh', max_iter=2000, solver='adam', alpha=0.001, hidden_layer_sizes=(60, 30))  # ovr:一对多策略
            # classifier = svm.SVC(C=3, kernel='linear', degree=2, gamma=1, probability=True, decision_function_shape='ovr')
            classifier.fit(x_train, targets_train)
            Y_test = classifier.predict(x_test)

            p[k] = precision_score(targets_test, Y_test, average='macro')
            r[k] = recall_score(targets_test, Y_test, average='macro')
            f1score[k] = f1_score(targets_test, Y_test, average='macro')
            ACC[k] = accuracy_score(targets_test, Y_test)
            ARS[k] = metrics.adjusted_rand_score(targets_test, Y_test)
            DBI[k] = metrics.davies_bouldin_score(x_test, Y_test)
            SS[k] = metrics.silhouette_score(x_test, Y_test)
            MCC[k] = matthews_corrcoef(targets_test, Y_test)


            n_class = max(targets_train)+1
            y_one_hot = label_binarize(targets_test, np.arange(n_class))  # 装换成类似二进制的编码
            y_score = classifier.predict_proba(x_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
            AUC[k] = metrics.auc(fpr, tpr)
            pr, re, thresholds = precision_recall_curve(y_one_hot.ravel(), y_score.ravel())
            PR_AUC[k] = auc(re, pr)

            name = path + f_name +  'mlp.txt'
            f = open(name, "a")
            print('第%d五倍交叉，交叉%d' % (n, k), file=f)
            print('Precision : ', p[k], file=f)
            print('Recall : ', r[k], file=f)
            print('f1score : ', f1score[k], file=f)
            print('ACC : ', ACC[k], file=f)
            print('ARI : ', ARS[k], file=f)
            print('DBI : ', DBI[k], file=f)
            print('silhouette_width : ', SS[k], file=f)
            print('MCC : ', MCC[k], file=f)
            print('AUC : ', AUC[k], file=f)
            print('PR_AUC : ', PR_AUC[k], file=f)
            f.close()
            k = k + 1

        print('Precision : ', np.mean(p))
        print('Recall : ', np.mean(r))
        print('f1score : ', np.mean(f1score))
        print('ACC : ', np.mean(ACC))
        print('ARI : ', np.mean(ARS))
        print('DBI : ', np.mean(DBI))
        print('silhouette_width : ', np.mean(SS))
        print('MCC : ', np.mean(MCC))
        print('AUC : ', np.mean(AUC))
        print('PR_AUC : ', np.mean(PR_AUC))

        name = path + f_name + 'mlp.txt'
        f = open(name, "a")
        print('第%d五倍交叉结果' % n, file=f)
        print('Precision : ', np.mean(p), file=f)
        print('Recall : ', np.mean(r), file=f)
        print('f1score : ', np.mean(f1score), file=f)
        print('ACC : ', np.mean(ACC), file=f)
        print('ARI : ', np.mean(ARS), file=f)
        print('DBI : ', np.mean(DBI), file=f)
        print('silhouette_width : ', np.mean(SS), file=f)
        print('MCC : ', np.mean(MCC), file=f)
        print('AUC : ', np.mean(AUC), file=f)
        print('PR_AUC : ', np.mean(PR_AUC), file=f)
        f.close()

        s_p = s_p + np.mean(p)
        s_r = s_r + np.mean(r)
        s_f1score = s_f1score + np.mean(f1score)
        s_ACC = s_ACC + np.mean(ACC)
        s_AUC = s_AUC + np.mean(AUC)
        s_PR_AUC = s_PR_AUC + np.mean(PR_AUC)
        s_ARS = s_ARS + np.mean(ARS)
        s_DBI = s_DBI + np.mean(DBI)
        s_SS = s_SS + np.mean(SS)
        s_MCC = s_MCC + np.mean(MCC)

        name = path + f_name + 'mlp_n_mean.txt'
        f = open(name, "a")
        print('第%d五倍交叉平均结果' % n, file=f)
        print('Precision : ', s_p / (n + 1), file=f)
        print('Recall : ', s_r / (n + 1), file=f)
        print('f1score : ', s_f1score / (n + 1), file=f)
        print('ACC : ', s_ACC / (n + 1), file=f)
        print('ARI : ', s_ARS / (n + 1), file=f)
        print('DBI : ', s_DBI / (n + 1), file=f)
        print('silhouette_width : ', s_SS / (n + 1), file=f)
        print('MCC : ', s_MCC / (n + 1), file=f)
        print('AUC : ', s_AUC / (n + 1), file=f)
        print('PR_AUC : ', s_PR_AUC / (n + 1), file=f)
        f.close()
    name = path + f_name +'mlp_mean.txt'
    f = open(name, "a")

    print('Precision : ', s_p / (n + 1), file=f)
    print('Recall : ', s_r / (n + 1), file=f)
    print('f1score : ', s_f1score / (n + 1), file=f)
    print('ACC : ', s_ACC / (n + 1), file=f)
    print('ARI : ', s_ARS / (n + 1), file=f)
    print('DBI : ', s_DBI / (n + 1), file=f)
    print('silhouette_width : ', s_SS / (n + 1), file=f)
    print('MCC : ', s_MCC / (n + 1), file=f)
    print('AUC : ', s_AUC / (n + 1), file=f)
    print('PR_AUC : ', s_PR_AUC / (n + 1), file=f)
    f.close()