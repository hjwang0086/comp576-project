import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from copy import deepcopy
from utils import pickle_load

def get_retrieval_imgs(nn_inds_path, img_indices=[0,1],
        query_file="data/oxford5k/test_query.txt", 
        gallery_file="data/oxford5k/test_gallery.txt",
        gnd_file="./data/oxford5k/gnd_roxford5k.pkl",
        save_name=None):
    
    # Get query and gallery images
    with open(query_file) as fid:
        query_lines  = fid.read().splitlines()
    with open(gallery_file) as fid:
        gallery_lines = fid.read().splitlines()

    # Get nn_inds and gnd
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
    num_samples, _ = cache_nn_inds.shape

    gnd_data = pickle_load(gnd_file)["gnd"]

    # Exclude junk images (Medium)
    medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
    for i in range(num_samples):
        junk_ids = gnd_data[i]['junk']
        all_ids = medium_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        medium_nn_inds[i] = new_ids

    # update ground truth using easy + hard
    gnd_data = {i: set(gnd_data[i]["easy"]) | set(gnd_data[i]["hard"]) for i in range(num_samples)}

    # Plot figure
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    fig.suptitle("Reranking list", y=0.96)
    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])

    for x, i in enumerate(img_indices):
        query_img = "data/oxford5k/" + query_lines[i].split(",")[0]
        axes[x][0].imshow(Image.open(query_img).resize((100, 100)))
        axes[x][0].set_ylabel(query_img.split("/")[-1], fontsize=8) # for debug
        axes[x][0].set_title(f"Index = {i}", fontsize=6, y=0.96)
        axes[1][0].set_xlabel("Query")
        
        for y in range(5):
            nn_inds = medium_nn_inds[i][y]
            gallery_img = "data/oxford5k/" + gallery_lines[nn_inds].split(",")[0]
            axes[x][y+1].imshow(Image.open(gallery_img).resize((100, 100)))
            axes[x][y+1].set_ylabel(gallery_img.split("/")[-1], fontsize=8)
            axes[x][y+1].set_title(f"Index {nn_inds}, Correct = {nn_inds in gnd_data[i]}", fontsize=6, y=0.96)
            axes[1][y+1].set_xlabel(f"Ranking #{y+1}")

    if not save_name:
        save_name = "retrieval_" + nn_inds_path.split("/")[-1].split(".")[0] + ".png"
    plt.savefig(save_name)
    plt.close()
    print(f"Figure \"{save_name}\" saved!")


if __name__ == "__main__":
    img_indices = [11, 65]

    nn_inds_path = "./data/oxford5k/nn_inds_r50_gldv2.pkl"
    get_retrieval_imgs(nn_inds_path, img_indices=img_indices, save_name="retrieval_no_rrt.png") 

    nn_inds_path = "./logs/gldv2_r50_RRT/1/medium_nn_inds.pkl"
    get_retrieval_imgs(nn_inds_path, img_indices=img_indices, save_name="retrieval_rrt.png") 

    nn_inds_path = "./logs/self-supervised/gldv2_r50_finetuneBCE/1/medium_nn_inds.pkl"
    get_retrieval_imgs(nn_inds_path, img_indices=img_indices, save_name="retrieval_rrt_enhanced.png") 

    # 11: T F F
    # 16: T T F
    # 26: T T F
    # 27: T F T
    # 35 36 37 38 39: T T F
    # 45 46 47 48 49: T T F
    # 60: T T F
    # 64: F T F
    # 65: F F F
    # 66: F F F