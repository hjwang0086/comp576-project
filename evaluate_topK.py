import matplotlib.pyplot as plt
import numpy as np
import torch

from copy import deepcopy

from utils import pickle_load
from utils.revisited import compute_metrics


def evaluate_roxf_medium_map(nn_inds_path, gnd_path, top_k=100):
    """
    ref: utils.metrics.mean_average_precision_revisited_rerank
    """

    # Read file
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
    gnd = pickle_load(gnd_path)

    num_samples, _ = cache_nn_inds.size()

    # Exclude the junk images as in DELG
    medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
    for i in range(num_samples):
        junk_ids = gnd['gnd'][i]['junk']
        all_ids = medium_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        medium_nn_inds[i] = new_ids
    medium_nn_inds = torch.from_numpy(medium_nn_inds)

    # Evaluate on current output
    ranks = deepcopy(medium_nn_inds)
    medium_curr = compute_metrics('revisited', ranks.T, gnd['gnd'], kappas=[])

    # Get global optimal solution
    best_ranks = deepcopy(medium_nn_inds.cpu().data.numpy())
    top, remain = best_ranks[:,:top_k], best_ranks[:,top_k:]
    for i in range(num_samples):
        label_ids = gnd['gnd'][i]["easy"] + gnd['gnd'][i]["hard"]
        all_ids = top[i]
        pos = np.in1d(all_ids, label_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([all_ids[pos], all_ids[neg]])
        top[i] = new_ids

    best_ranks = np.concatenate([top, remain], axis=1)
    best_ranks = torch.from_numpy(best_ranks)
    medium_best = compute_metrics('revisited', best_ranks.T, gnd['gnd'], kappas=[])

    # Output
    print("Current mAP:", float(medium_curr['M_map']))
    print("Optimal mAP:", float(medium_best['M_map']))

    return float(medium_curr['M_map']) / 100, float(medium_best['M_map']) / 100


if __name__ == "__main__":
    
    top_ks = range(100, 2001, 100)
    best_mAP_list = []
    for top_k in top_ks:
        _, best_mAP = evaluate_roxf_medium_map(
            nn_inds_path = "./data/oxford5k/nn_inds_r50_gldv2.pkl",
            gnd_path = "./data/oxford5k/gnd_roxford5k.pkl", 
            top_k=top_k
        )

        best_mAP_list.append(best_mAP)

    plt.scatter(top_ks, best_mAP_list, color="gray", label="upper bound")
    plt.plot(top_ks, best_mAP_list, color="gray")
    plt.xlabel("Top k")
    plt.ylabel("mAP")
    plt.title("ROxf mAP Curve")
    plt.legend(loc="lower right")
    plt.savefig("optimal_map_curve.png")
    plt.close()