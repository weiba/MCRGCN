
import numpy as np
from math import sqrt
from sklearn import preprocessing
import scipy.io as sio
a1=0.41
a2=0.35
a3=0.42
# 用下面函数可加速算法运行
def corr_x_y(x: np.ndarray, y: np.ndarray):
    """
    Calculate the correlation coefficient between matrix x and y.
    where x in n \times k, y in m \times k
    :param x: matrix, np.ndarray, shape (n, k)
    :param y: matrix, np.ndarray, shape (m, k)
    :return: correlation coefficient, np.ndarray, shape (n, m)
    """
    assert x.shape[1] == y.shape[1], "Different shape!"
    x = x - np.mean(x, axis=1).reshape((-1, 1))
    y = y - np.mean(y, axis=1).reshape((-1, 1))
    lxy = np.dot(x, y.T)
    lxx = np.diag(np.dot(x, x.T)).reshape((-1, 1))
    lyy = np.diag(np.dot(y, y.T)).reshape((1, -1))
    std_x_y = np.dot(np.sqrt(lxx), np.sqrt(lyy))
    corr = lxy / std_x_y
    return corr


def cos_similarity(x: np.ndarray, y: np.ndarray):
    """
    Calculate the cos similarity between matrix x and y.
    where x in n \times k, y in m \times k
    :param x: matrix, np.ndarray, shape (n, k)
    :param y: matrix, np.ndarray, shape (m, k)
    :return: cos similarity, np.ndarray, shape (n, m)
    """
    assert x.shape[1] == y.shape[1], "Different shape!"
    xy = np.dot(x, y.T)
    module_x = np.sqrt(np.diag(np.dot(x, x.T))).reshape((-1, 1))
    module_y = np.sqrt(np.diag(np.dot(y, y.T))).reshape((1, -1))
    module_x_y = np.dot(module_x, module_y)
    simi = xy / module_x_y
    return simi


def euclidean_distance(x: np.ndarray, y: np.ndarray):
    """
    Calculate the euclidean distance between matrix x and y.
    where x in n \times k, y in m \times k
    :param x: matrix, np.ndarray, shape (n, k)
    :param y: matrix, np.ndarray, shape (m, k)
    :return: euclidean distance, np.ndarray, shape (n, m)
    """
    assert x.shape[1] == y.shape[1], "Different shape!"
    xy = np.dot(x, y.T)
    xx = np.diag(np.dot(x, x.T)).reshape((-1, 1))
    yy = np.diag(np.dot(y, y.T)).reshape((1, -1))
    dist = xx + yy - 2*xy
    dist = np.sqrt(dist)
    return dist


if __name__ == '__main__':

    data = sio.loadmat('GBM.mat')
    # np.ndarray shape: 213, 12042
    features = data['GBM_Gene_Expression'].T
    print(features.shape)
    Methy_features = data['GBM_Methy_Expression'].T
    Mirna_features = data['GBM_Mirna_Expression'].T
    # np.ndarray shape: 213, 1
    targets = data['GBM_clinicalMatrix']
    # np.ndarray shape: 213, 1
    indexes = data['GBM_indexes']

    features = preprocessing.scale(features)
    Methy_features = preprocessing.scale(Methy_features)
    Mirna_features = preprocessing.scale(Mirna_features)

    z1 =abs(corr_x_y(x=features, y=features))
    normalize_corr1 = np.where(z1 > a1, 1, 0)
    z2= abs(corr_x_y(x= Methy_features, y= Methy_features))
    normalize_corr2 = np.where(z2 > a2, 1, 0)
    z3= abs(corr_x_y(x=Mirna_features, y=Mirna_features))
    normalize_corr3 = np.where(z3 > a3, 1, 0)
    # save data
    f_name = './data/'
    # name1 = f_name + 'BRCA_matrix.mat'
    name1= f_name + 'GBM_gene_matrix.mat'
    sio.savemat(name1, {'normalize_corr':normalize_corr1})
    name2 = f_name + 'GBM_methy_matrix.mat'
    sio.savemat(name2, {'normalize_corr': normalize_corr2})
    name3 = f_name + 'GBM_mirna_matrix.mat'
    sio.savemat(name3, {'normalize_corr': normalize_corr3})