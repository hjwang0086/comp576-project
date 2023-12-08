import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # surpass warning

import _init_paths
import os.path as osp
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import sacred
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from numpy import linalg as LA
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from utils import pickle_load, pickle_save
from utils.data.delf import datum_io


def get_positives(gnd_path):
    mat = loadmat(gnd_path)
    matStruct = mat['dbStruct'].item()

    utmDb = matStruct[2].T
    utmQ = matStruct[4].T

    posDistThr = 25 # no need fix currently, otherwise complicate implementation

    # fit kNN
    # https://github.com/Nanne/pytorch-NetVlad/blob/master/tokyo247.py#L105
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(utmDb)
    distances, positives = knn.radius_neighbors(utmQ, posDistThr)

    return positives


def evaluate_recalls(gt, predictions, n_values=[1,5,10,20]):
    # https://github.com/Nanne/pytorch-NetVlad/blob/master/main.py#L205
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / len(gt)

    recalls = {} # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    return recalls


ex = sacred.Experiment('Prepare Top-K (VPR)')
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    dataset_name = 'tokyo247'
    feature_name = 'r50_gldv2'
    gnd_name = 'tokyo247.mat'

    if dataset_name == 'tokyo247':
        real_name = 'Tokyo247'
    elif dataset_name == "pitts30k":
        real_name = 'Pittsburgh250k'
    data_dir = osp.join('data', real_name)

    save_nn_inds = True
    

@ex.automain
def main(data_dir, feature_name, gnd_name, save_nn_inds):
    with open(osp.join(data_dir, 'test_query.txt')) as fid:
        query_lines   = fid.read().splitlines()
    with open(osp.join(data_dir, 'test_gallery.txt')) as fid:
        gallery_lines = fid.read().splitlines()

    query_feats = []
    for i in tqdm(range(len(query_lines))):
        name = osp.splitext(osp.basename(query_lines[i].split(',')[0]))[0]
        path = osp.join(data_dir, 'delg_'+feature_name, name+'.delg_global')
        query_feats.append(datum_io.ReadFromFile(path))
        
    query_feats = np.stack(query_feats, axis=0)
    query_feats = query_feats/LA.norm(query_feats, axis=-1)[:,None]

    index_feats = []
    for i in tqdm(range(len(gallery_lines))):
        name = osp.splitext(osp.basename(gallery_lines[i].split(',')[0]))[0]
        path = osp.join(data_dir, 'delg_'+feature_name, name+'.delg_global')
        index_feats.append(datum_io.ReadFromFile(path))
        
    index_feats = np.stack(index_feats, axis=0)
    index_feats = index_feats/LA.norm(index_feats, axis=-1)[:,None]

    sims = np.matmul(query_feats, index_feats.T)

    nn_inds  = np.argsort(-sims, -1)
    nn_dists = deepcopy(sims)
    for i in range(query_feats.shape[0]):
        for j in range(index_feats.shape[0]):
            nn_dists[i, j] = sims[i, nn_inds[i, j]]

    if save_nn_inds:
        output_path = osp.join(data_dir, 'nn_inds_%s.pkl'%feature_name)
        pickle_save(output_path, nn_inds)

    # evaluate gnd_data
    gnd_path = osp.join(data_dir, gnd_name)
    positives = get_positives(gnd_path)
    evaluate_recalls(positives, nn_inds)