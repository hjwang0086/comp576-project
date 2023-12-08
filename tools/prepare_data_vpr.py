"""
create test_query.txt, test_gallery.txt, gnd_*.pkl
"""

import _init_paths  # to know local directory
import os
import os.path as osp

from tqdm import tqdm
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
from shutil import copyfile
from PIL import Image

from utils import pickle_save

def parse_struct(filepath):
    mat = loadmat(filepath)
    matStruct = mat['dbStruct'].item()
    dbImage = [f[0].item() for f in matStruct[1]]
    qImage = [f[0].item() for f in matStruct[3]]

    return dbImage, qImage

def write_metadata(dbImage, qImage, data_dir, query_root, gallery_root, rename_query=False):
    query_lines, gallery_lines = [], []

    # write query line
    for i, img_file in enumerate(qImage, 1):
        print(f"Extract query {i}/{len(qImage)}...", end="\r")
        if rename_query:
            img_file = img_file.split("/")
            img_file = img_file[:-1] + ["query_" + img_file[-1]]
            img_file = "/".join(img_file)
        img_path = osp.join(data_dir, query_root, img_file)
        img = Image.open(img_path)
        width, height = img.size

        line = ",".join([img_file, "0", str(width), str(height)]) # no label
        query_lines.append(line)
        
    print()
    
    with open(osp.join(data_dir, "test_query.txt"), "w") as f:
        f.write("\n".join(query_lines))

    # write gallery line
    for i, img_file in enumerate(dbImage, 1):
        print(f"Extract gallery {i}/{len(dbImage)}...", end="\r")
        img_path = osp.join(data_dir, gallery_root, img_file)

        # case for Tokyo247 that metadata shows .jpg while database shows .png
        if not osp.isfile(img_path) and img_file[-4:] == ".jpg":
            img_file = img_file[:-4] + ".png"
            img_path = osp.join(data_dir, gallery_root, img_file)

        img = Image.open(img_path)
        width, height = img.size

        line = ",".join([img_file, "0", str(width), str(height)]) # no label
        gallery_lines.append(line)
        
    print()
    
    with open(osp.join(data_dir, "test_gallery.txt"), "w") as f:
        f.write("\n".join(gallery_lines))


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

if __name__ == "__main__":

    ############
    # tokyo247 #
    ############
    # gnd_name = "data/Pittsburgh250k/datasets/tokyo247.mat"
    # data_dir = "data/Tokyo247"
    # query_root = "queries/247query_subset_v2"
    # gallery_root = "database_gsv_vga"
    # gnd_pkl_file = "gnd_tokyo247.pkl"

    # # parse
    # dbImage, qImage = parse_struct(gnd_name)
    # write_metadata(dbImage, qImage, data_dir, query_root, gallery_root)

    # # copy gnd file
    # dst = osp.join(data_dir, osp.basename(gnd_name))
    # copyfile(gnd_name, dst)

    # # create pkl file (has knn info)
    # positives = get_positives(dst)
    # out = {}
    # out["gnd"] = [list(x) for x in positives]
    # pickle_save(osp.join(data_dir, gnd_pkl_file), out)

    ############
    # pitts30k #
    ############

    gnd_name = "data/Pittsburgh250k/datasets/pitts30k_test.mat"
    data_dir = "data/Pittsburgh250k"
    query_root = "queries_real_renamed"
    gallery_root = "."
    gnd_pkl_file = "gnd_pitts30k.pkl"

    # copy data to new query root and rename it
    query_root_origin = "queries_real"
    if not osp.isdir(osp.join(data_dir, query_root)):
        os.system("cp -r {} {}".format(osp.join(data_dir, query_root_origin), osp.join(data_dir, query_root)))

        for i in range(9):
            sub_folder = str(i).zfill(3)
            for img_file in os.listdir(osp.join(data_dir, query_root, sub_folder)):
                img_path = osp.join(data_dir, query_root, sub_folder, img_file)
                new_name = "query_" + img_file
                new_name = osp.join(data_dir, query_root, sub_folder, new_name)
                os.rename(img_path, new_name)

    # parse
    dbImage, qImage = parse_struct(gnd_name)
    write_metadata(dbImage, qImage, data_dir, query_root, gallery_root, rename_query=True)

    # copy ground truth file
    dst = osp.join(data_dir, osp.basename(gnd_name))
    copyfile(gnd_name, dst)

    # create pkl file (has knn info)
    positives = get_positives(dst)
    out = {}
    out["gnd"] = [list(x) for x in positives]
    pickle_save(osp.join(data_dir, gnd_pkl_file), out)