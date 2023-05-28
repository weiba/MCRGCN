
from sklearn.cluster import KMeans
import numpy as np
import torch
import scipy.sparse as sp
def clusters(embeds):
    device = torch .device('cuda:0' if torch.cuda.is_available() else 'cpu')
    embeds=embeds.cuda().data.cpu().numpy()
    kmeans = KMeans(n_clusters=4, max_iter=300, n_init=10).fit(embeds)
    lables =  kmeans.labels_

    pos = np.zeros((lables.shape[0], lables.shape[0]))
    for i in range(lables.shape[0]):
        for j in range(lables.shape[0]):
            if lables[i]==lables[j]:
                pos[i][j]=1
            else:
                pos[i][j]=0
    pos = sp.coo_matrix(pos)
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    pos=pos.to_dense().to(device)
    # print(pos)
    return pos





def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch .from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch .from_numpy(sparse_mx.data)
    shape = torch .Size(sparse_mx.shape)
    return torch .sparse.FloatTensor(indices, values, shape)