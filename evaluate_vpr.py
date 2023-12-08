import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # surpass warning

from copy import deepcopy
from functools import partial
from pprint import pprint
import os.path as osp

import numpy as np
import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load, pickle_save
from utils.data.dataset_ingredient import data_ingredient, get_loaders



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

ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    visdom_port = None
    visdom_freq = 20
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('logs', 'temp')
    resume = None
    seed = 0
    desc_name = 'r50_gldv2'
    output_path = '.'
    gnd_path = 'data/Tokyo247/tokyo247.mat'
    use_bottleneck = False # whether FC layer (128->256) is added before MultiHeadAttn
    use_duplicate = False # whether duplicate token channel before MultiHeadAttn

@ex.automain
def main(cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, resume, desc_name, output_path, gnd_path,
            use_bottleneck, use_duplicate):
    
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders(desc_name=desc_name)

    torch.manual_seed(seed)
    model = get_model(use_bottleneck=use_bottleneck, use_duplicate=use_duplicate)
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state'], strict=True)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
    num_samples, top_k = cache_nn_inds.size()
    top_k = min(100, top_k)
    
    ####################################
    # write eval function in main body #
    ###################################
    
    # (1) utils.training.evaluate
    query_loader = loaders.query
    gallery_loader = loaders.gallery

    model.eval()
    query_global, query_local, query_mask, query_scales, query_positions = [], [], [], [], []
    gallery_global, gallery_local, gallery_mask, gallery_scales, gallery_positions = [], [], [], [], []

    with torch.no_grad():
        for entry in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80):
            q_global, q_local, q_mask, q_scales, q_positions, _, _ = entry
            query_global.append(q_global.cpu())
            query_local.append(q_local.cpu())
            query_mask.append(q_mask.cpu())
            query_scales.append(q_scales.cpu())
            query_positions.append(q_positions.cpu())

        query_global    = torch.cat(query_global, 0)
        query_local     = torch.cat(query_local, 0)
        query_mask      = torch.cat(query_mask, 0)
        query_scales    = torch.cat(query_scales, 0)
        query_positions = torch.cat(query_positions, 0)

        for entry in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
            g_global, g_local, g_mask, g_scales, g_positions, _, _ = entry
            gallery_global.append(g_global.cpu())
            gallery_local.append(g_local.cpu())
            gallery_mask.append(g_mask.cpu())
            gallery_scales.append(g_scales.cpu())
            gallery_positions.append(g_positions.cpu())
            
        gallery_global    = torch.cat(gallery_global, 0)
        gallery_local     = torch.cat(gallery_local, 0)
        gallery_mask      = torch.cat(gallery_mask, 0)
        gallery_scales    = torch.cat(gallery_scales, 0)
        gallery_positions = torch.cat(gallery_positions, 0)

        torch.cuda.empty_cache()
        # evaluate_function = partial(mean_average_precision_revisited_rerank, model=model, cache_nn_inds=cache_nn_inds,
        #     query_global=query_global, query_local=query_local, query_mask=query_mask, query_scales=query_scales, query_positions=query_positions, 
        #     gallery_global=gallery_global, gallery_local=gallery_local, gallery_mask=gallery_mask, gallery_scales=gallery_scales, gallery_positions=gallery_positions, 
        #     ks=recall, 
        #     gnd=query_loader.dataset.gnd_data, output_path=output_path
        # )
        # metrics = evaluate_function()

        # (2) mimic utils.metrics.mean_average_precision_revisited_rerank
        os.makedirs(output_path, exist_ok=True)
        query_global    = query_global.to(device)
        query_local     = query_local.to(device)
        query_mask      = query_mask.to(device)
        query_scales    = query_scales.to(device)
        query_positions = query_positions.to(device)
        
        scores = []
        for i in tqdm(range(top_k)):
            nnids = cache_nn_inds[:, i]
            index_global    = gallery_global[nnids]
            index_local     = gallery_local[nnids]
            index_mask      = gallery_mask[nnids]
            index_scales    = gallery_scales[nnids]
            index_positions = gallery_positions[nnids]

            # batchify the inference to avoid CUDA OOM
            b_size = query_global.shape[0]
            new_b_size = 32
            current_scores = []
            for j in range(0, b_size // new_b_size + 1):
                start, end = j*new_b_size, min((j+1)*new_b_size, b_size)
                if start >= end: break
                current_batch_scores = model(
                    query_global[start:end], 
                    query_local[start:end], 
                    query_mask[start:end], 
                    query_scales[start:end], 
                    query_positions[start:end],
                    index_global[start:end].to(device),
                    index_local[start:end].to(device),
                    index_mask[start:end].to(device),
                    index_scales[start:end].to(device),
                    index_positions[start:end].to(device))
                current_scores.extend(current_batch_scores.cpu().data)
            
            current_scores = torch.tensor(current_scores)
            scores.append(current_scores)
            
        scores = torch.stack(scores, -1)
        closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
        closest_indices = torch.gather(cache_nn_inds, -1, indices)
        ranks = deepcopy(cache_nn_inds)
        ranks[:, :top_k] = deepcopy(closest_indices)
        ranks = ranks.cpu().data.numpy()
        
        pickle_save(osp.join(output_path, 'reranked_nn_inds.pkl'), ranks)
        
        # evaluate gnd_data
        positives = get_positives(gnd_path)
        evaluate_recalls(positives, ranks)