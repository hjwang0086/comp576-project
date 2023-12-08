import os, csv
import pandas as pd

# create train_all_clean.txt
df = pd.read_csv("data/gldv2/train/train_clean.csv")
all_images = []
for image_str in df["images"].values.tolist():
    images = image_str.split(" ")
    all_images.extend(images)
print("Total # images:", len(all_images))

with open("data/gldv2/train_all_clean.txt", "w") as fout:
    for i, image in enumerate(all_images):
        f1, f2, f3 = image[0], image[1], image[2]
        fout.writelines(f"train/{f1}/{f2}/{f3}/{image}.jpg,_,_,_")
        if i != len(all_images) - 1:
            fout.writelines("\n")