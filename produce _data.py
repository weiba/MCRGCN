
import numpy as np
import time
import pandas as pd
import os
import sys
import scipy.io as sio
from math import sqrt
import scipy.sparse as sp
from sklearn.model_selection import KFold


A1=sio.loadmat('./data/GBM_gene_matrix.mat')
A2=sio.loadmat('./data/GBM_methy_matrix.mat')
A3=sio.loadmat('./data/GBM_mirna_matrix.mat')
A1=A1['normalize_corr']
A2=A2['normalize_corr']
A3=A3['normalize_corr']

edges1=sp.coo_matrix(A1)
print(edges1)
print(edges1.data.shape[0])
file_write_obj = open("./data/edges_gene_gbm.csv", 'w+')
for id in np.arange(edges1.data.shape[0]):
    file_write_obj.writelines(np.str(edges1.row[id]))
    file_write_obj.write(',')
    file_write_obj.writelines(np.str(edges1.col[id]))
    file_write_obj.write('\n')
file_write_obj.close()

edges2=sp.coo_matrix(A2)
print(edges2)
print(edges2.data.shape[0])
file_write_obj = open("./data/edges_methy_gbm.csv", 'w+')
for id in np.arange(edges2.data.shape[0]):
    file_write_obj.writelines(np.str(edges2.row[id]))
    file_write_obj.write(',')
    file_write_obj.writelines(np.str(edges2.col[id]))
    file_write_obj.write('\n')
file_write_obj.close()

edges3=sp.coo_matrix(A3)
print(edges3)
print(edges3.data.shape[0])
file_write_obj = open("./data/edges_mirna_gbm.csv", 'w+')
for id in np.arange(edges3.data.shape[0]):
    file_write_obj.writelines(np.str(edges3.row[id]))
    file_write_obj.write(',')
    file_write_obj.writelines(np.str(edges3.col[id]))
    file_write_obj.write('\n')
file_write_obj.close()
