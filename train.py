import torch

from torch_geometric.data import Data
import torch.nn.functional as fun
import torch.nn as nn
from sklearn import metrics
import numpy as np
import scipy.sparse as sp
from model.heco import HeCo
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, auc, precision_recall_curve, \
    matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from model.contrast import Contrast
from sklearn import svm
import scipy.io as sio
from sklearn import preprocessing
import torch.backends.cudnn as cudnn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import produce_adjacent_matrix
# print(produce_adjacent_matrix.a)
# 建议写成类、函数等模块化程序

data = sio.loadmat('GBM.mat')
features = data['GBM_Gene_Expression'].T
Methy_features = data['GBM_Methy_Expression'].T
Mirna_features = data['GBM_Mirna_Expression'].T
labels = data['GBM_clinicalMatrix']
indexes = data['GBM_indexes']
features= preprocessing.scale(features)
Methy_features=preprocessing.scale(Methy_features)
Mirna_features=preprocessing.scale(Mirna_features)


labels = labels.reshape(labels.shape[0])
path = "data/"
cites1= path + "edges_gene_gbm.csv"
cites2= path + "edges_methy_gbm.csv"
cites3= path + "edges_mirna_gbm.csv"
# 索引字典，转换到从0开始编码
index_gene_dict= dict()

edge_gene_index = []
draw_gene_edge_index = []

for i in range(indexes.shape[0]):
    index_gene_dict[int(indexes[i])] = len(index_gene_dict)
    print(index_gene_dict)

with open(cites1, "r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.split(',')
        edge_gene_index.append([index_gene_dict[int(start)], index_gene_dict[int(end)]])
        #edge_index.append([index_dict[int(end)], index_dict[int(start)]])

print(edge_gene_index)

index_methy_dict= dict()
edge_methy_index = []
draw_methy_edge_index = []

for i in range(indexes.shape[0]):
    index_methy_dict[int(indexes[i])] = len(index_methy_dict)
    print(index_methy_dict)

with open(cites2, "r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.split(',')
        edge_methy_index.append([index_methy_dict[int(start)], index_methy_dict[int(end)]])
        #edge_index.append([index_dict[int(end)], index_dict[int(start)]])

print(edge_methy_index)

index_mirna_dict= dict()
edge_mirna_index = []
draw_mirna_edge_index = []

for i in range(indexes.shape[0]):
    index_mirna_dict[int(indexes[i])] = len(index_mirna_dict)
    print(index_mirna_dict)

with open(cites3, "r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.split(',')
        edge_mirna_index.append([index_mirna_dict[int(start)], index_mirna_dict[int(end)]])
        #edge_index.append([index_dict[int(end)], index_dict[int(start)]])

print(edge_mirna_index)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch .from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch .from_numpy(sparse_mx.data)
    shape = torch .Size(sparse_mx.shape)
    return torch .sparse.FloatTensor(indices, values, shape)



labels = torch.LongTensor(labels)
features1=features
features2=Methy_features
features3=Mirna_features

# features1=features+np.random.normal(0, 1, size=(features.shape[0],features.shape[1]))
# features2=features
features1 = torch.FloatTensor(features1)
features2 = torch.FloatTensor(features2)
features3 = torch.FloatTensor(features3)
edge_gene_index = torch.LongTensor(edge_gene_index).t()
edge_methy_index = torch.LongTensor(edge_methy_index).t()
edge_mirna_index = torch.LongTensor(edge_mirna_index).t()
print(edge_gene_index)
print(edge_methy_index)
print(edge_mirna_index)

# 训练
# 固定种子
seed =1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

mask = torch.randperm(len(index_gene_dict))
print('.........')
print(mask)
print('.......')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cora1 = Data(x=features1, edge_index=edge_gene_index.contiguous(), y=labels).to(device)
cora2 = Data(x=features2, edge_index=edge_methy_index.contiguous(), y=labels).to(device)
cora3 = Data(x=features3, edge_index=edge_mirna_index.contiguous(), y=labels).to(device)

p_mean = np.zeros(10)
r_mean = np.zeros(10)
f1score_mean = np.zeros(10)
ACC_mean = np.zeros(10)
ARS_mean = np.zeros(10)
MCC_mean = np.zeros(10)
AUC_mean = np.zeros(10)
PR_AUC_mean = np.zeros(10)
DBI_mean = np.zeros(10)
SS_mean = np.zeros(10)
k = 5
dict = dict()
f_name = './results/'
for n in range(10):
    p = np.zeros(5)
    r = np.zeros(5)
    f1score = np.zeros(5)
    ACC = np.zeros(5)
    ARS = np.zeros(5)
    MCC = np.zeros(5)
    AUC = np.zeros(5)
    PR_AUC = np.zeros(5)
    DBI = np.zeros(5)
    SS = np.zeros(5)
    m = 0
    survial_test_lable = []
    survial_test_index = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=n*n+1)
    for train_mask, test_mask in kfold.split(mask):


        pos = np.zeros((cora1.y[train_mask].shape[0], cora1.y[train_mask].shape[0]))
        for i in range(cora1.y[train_mask].shape[0]):
            for j in range(cora1.y[train_mask].shape[0]):
                if cora1.y[train_mask][i] == cora1.y[train_mask][j]:
                    pos[i][j] = 1
                else:
                    pos[i][j] = 0
        print(pos)
        pos = sp.coo_matrix(pos)
        pos = sparse_mx_to_torch_sparse_tensor(pos)
        pos = pos.to_dense().to(device)
        print(pos)
        print(features2.shape[1])
        model = HeCo(features1.shape[1],features2.shape[1],features3.shape[1]).to(device)
        print(features3.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion= Contrast(128, 0.5, 0.5).to(device)
        for epoch in range(120):
            model.train()
            optimizer.zero_grad()
            z_ge,z_mp,z_sc=model(cora1,cora2,cora3)
            loss = criterion(z_ge[train_mask],z_mp[train_mask], z_sc[train_mask],pos)
            # revise - change: loss -> loss.item()
            print('epoch: %d loss: %.4f' % (epoch, loss.item()))
            loss.backward()
            optimizer.step()

        model.eval()
        embeds = model.get_embeds(cora1, cora2,cora3)
        embeds_train= embeds[train_mask]
        embeds_test = embeds[test_mask]
        targets_train=cora1.y[train_mask]
        targets_test =cora1.y[test_mask]
        embeds_train = embeds_train.cuda().data.cpu().numpy()
        embeds_test = embeds_test.cuda().data.cpu().numpy()
        targets_train = targets_train.cuda().data.cpu().numpy()
        targets_test = targets_test.cuda().data.cpu().numpy()
        print(targets_test)

        classifier = MLPClassifier(activation='tanh', max_iter=2000, solver='adam', alpha=0.001,
                                   hidden_layer_sizes=(60,30))  # ovr:一对多策略
        # classifier = svm.SVC(C=3, kernel='linear', degree=2, gamma=1, probability=True, decision_function_shape='ovr')
        # classifier = RandomForestClassifier()
        # classifier = KNeighborsClassifier(n_neighbors=4)
        classifier.fit(embeds_train, targets_train)
        Y_test = classifier.predict(embeds_test)

        # survial_lable = Y_test.tolist()
        # # survial_test_lable.extend(survial_lable)
        # survial_index = test_mask.tolist()
        # survial_test_index.extend(survial_index)
        # name = f_name + 'each' + '_' + 'survice_lable_gbm' + '.txt'
        # f = open(name, "a")
        # print(Y_test, file=f)
        # f.close()
        #
        # name = f_name + 'each' + '_' + 'survice_index_gbm' + '.txt'
        # f = open(name, "a")
        # print(test_mask, file=f)
        # f.close()

        p[m] = precision_score(targets_test, Y_test, average='macro')
        r[m] = recall_score(targets_test, Y_test, average='macro')
        f1score[m] = f1_score(targets_test, Y_test, average='macro')
        ACC[m] = accuracy_score(targets_test, Y_test)
        ARS[m] = metrics.adjusted_rand_score(targets_test, Y_test)
        MCC[m] = matthews_corrcoef(targets_test, Y_test)
        DBI[m] = metrics.davies_bouldin_score(embeds_test, Y_test)
        SS[m] = metrics.silhouette_score(embeds_test, Y_test)

        n_class = max(targets_train) + 1
        y_one_hot = label_binarize(targets_test,  classes=np.arange(4))  # 装换成类似二进制的编码
        y_score = classifier.predict_proba(embeds_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
        AUC[m] = metrics.auc(fpr, tpr)
        pr, re, thresholds = precision_recall_curve(y_one_hot.ravel(), y_score.ravel())
        PR_AUC[m] = auc(re, pr)

        name = f_name + 'each' + '_' + 'result_gbm_mlp' + '.txt'
        f = open(name, "a")
        print('第%d五倍交叉，交叉%d' % (n, m), file=f)
        print('Precision : ', p[m], file=f)
        print('Recall : ', r[m], file=f)
        print('f1score : ', f1score[m], file=f)
        print('ACC : ', ACC[m], file=f)
        print('ARI : ', ARS[m], file=f)
        print('MCC : ', MCC[m], file=f)
        print('AUC : ', AUC[m], file=f)
        print('PR_AUC : ', PR_AUC[m], file=f)
        print('silhouette_width : ', SS[m], file=f)
        print('DBI : ', DBI[m], file=f)
        f.close()
        m = m + 1




    name = f_name + 'each' + '_' + 'result' + '_n_mean_gbm_mlp.txt'
    f = open(name, "a")
    print('第%d五倍交叉平均结果' % n, file=f)
    print('Precision : ', np.mean(p), file=f)
    print('Recall : ', np.mean(r), file=f)
    print('f1score : ', np.mean(f1score), file=f)
    print('ACC : ', np.mean(ACC), file=f)
    print('ARI : ', np.mean(ARS), file=f)
    print('MCC : ', np.mean(MCC), file=f)
    print('AUC : ', np.mean(AUC), file=f)
    print('PR_AUC : ', np.mean(PR_AUC), file=f)
    print('DBI : ', np.mean(DBI), file=f)
    print('silhouette_width : ', np.mean(SS), file=f)
    f.close()
    p_mean[n] = np.mean(p)
    r_mean[n] = np.mean(r)
    f1score_mean[n] = np.mean(f1score)
    ACC_mean[n] = np.mean(ACC)
    ARS_mean[n] = np.mean(ARS)
    MCC_mean[n] = np.mean(MCC)
    AUC_mean[n] = np.mean(AUC)
    PR_AUC_mean[n] = np.mean(PR_AUC)
    DBI_mean[n] = np.mean(DBI)
    SS_mean[n] = np.mean(SS)

s_p_mean = np.mean(p_mean)
s_r_mean = np.mean(r_mean)
s_f1score_mean = np.mean(f1score_mean)
s_ACC_mean = np.mean(ACC_mean)
s_ARS_mean = np.mean(ARS_mean)
s_MCC_mean = np.mean(MCC_mean)
s_AUC_mean = np.mean(AUC_mean)
s_PR_AUC_mean = np.mean(PR_AUC_mean)
s_DBI_mean = np.mean(DBI_mean)
s_SS_mean = np.mean(SS_mean)

name = f_name + 'each' + '_' + 'result_mlp' + '_gbm_mean.txt'
f = open(name, "a")

print('十次五倍交叉平均结果', file=f)
print('Precision : ', s_p_mean, file=f)
print('Recall : ', s_r_mean, file=f)
print('f1score : ', s_f1score_mean, file=f)
print('ACC : ', s_ACC_mean, file=f)
print('ARI : ', s_ARS_mean, file=f)
print('MCC : ', s_MCC_mean, file=f)
print('AUC : ', s_AUC_mean, file=f)
print('PR_AUC : ', s_PR_AUC_mean, file=f)
print('DBI : ', s_DBI_mean, file=f)
print('silhouette_width : ', s_SS_mean, file=f)
f.close()









