# RRTv2: Improved Baselines with Transformer-based Image Retrieval and Reranking

The codebase is an improvement based on the [RRT repository](https://github.com/uvavision/RerankingTransformer/tree/main/RRT_GLD), with the following added features:

1. Detailed instructions on how to run the codebase.
2. Support for inferencing on Visual Place Recognition datasets such as Pitts30k and Tokyo 24/7.
3. Ability to use newly-trained DELG descriptors (requires ```is_tf2_exported=True```).
4. Support for training with higher dimensions of local features.
5. Support self-supervised pretraining on RRT.

Please note that the DELG training script has not been updated in this codebase yet. It will be incorporated in the future. For now, please run the DELG training script separately, referring to the original [DELG repository](https://github.com/tensorflow/models/tree/master/research/delf/delf/python/training).

---

## Training & Inference Pipelines

The whole training and inference pipelines consists of:

1. Train DELG backbone
2. Extract DELG features
3. Global retrieval
4. Train & inference RRT

### 1. Train DELG backbone

Follow these steps:

1. Navigate to the [DELG repository](https://github.com/tensorflow/models/tree/master/research/delf/delf/python/training).
2. Run the following code to train a DELG extractor, replacing `<model_name>` with the desired model name:

```
python train.py \
  --train_file_pattern=gldv2_dataset/tfrecord/train* \
  --validation_file_pattern=gldv2_dataset/tfrecord/validation* \
  --imagenet_checkpoint=resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  --dataset_version=gld_v2_clean \
  --logdir=gldv2_training/<model_name> \
  --delg_global_features --block3_strides=False \
  --max_iters=1500000

python model/export_local_and_global_model.py \
  --ckpt_path=gldv2_training/<model_name>/delf_weights \
  --export_path=gldv2_training/<model_name>/model \
  --delg_global_features --block3_strides=False
```

Once the training is completed, **create a hyperlink named `gldv2_training` to the codebase**. For example:

```
cd <rrt_root>
ln -s <delg_root>/gldv2_training .
```

### 2. Extract DELG features

Navigate to the directory of the codebase and execute the following commands to generate DELG features for the GLDv2 and ROxf datasets. Please ensure that the GLDv2 and ROxf datasets are stored in the `data` folder:

```
# GLDv2
python ~/delg/extract_features_gld.py \
  --delf_config_path parameters/config.pbtxt \
  --dataset_file_path data/gldv2/train.txt \
  --images_dir data/gldv2 --is_tf2_exported \
  --output_features_dir data/gldv2/delg_<model_name> \
  --model_path gldv2_training/<model_name>/model

# ROxf (query + gallery)
python ~/delg/extract_features.py \
  --delf_config_path parameters/config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set query --is_tf2_exported \
  --output_features_dir data/oxford5k/delg_<model_name> \
  --model_path gldv2_training/<model_name>/model

python ~/delg/extract_features.py \
  --delf_config_path parameters/config.pbtxt \
  --dataset_file_path data/oxford5k/gnd_roxford5k.mat \
  --images_dir data/oxford5k/jpg \
  --image_set index --is_tf2_exported \
  --output_features_dir data/oxford5k/delg_<model_name> \
  --model_path gldv2_training/<model_name>/model
```

### 3. Global retrieval

Global retrieval tries to generate nearest-neighbor indices based on the similarity scores of DELG features. Run the following code to generate the nearest-neighbor indices file:

```
python tools/prepare_topk_gldv2.py with feature_name=<model_name>
python tools/prepare_topk_revisited.py with dataset_name=oxford5k feature_name=<model_name> gnd_name=gnd_roxford5k.pkl
```

### 4. Train & inference RRT

Execute the following command, where `<path>` is the parent directory for the model and can be ignored:

```
python experiment.py -F logs/<path>/<model_name> with \
  dataset.gldv2_roxford_r50_gldv2 model.RRT max_norm=0.0 \
   desc_name=<model_name>
python evaluate_revisited.py with dataset.roxford_r50_gldv2 \
   model.RRT resume=logs/<path>/<model_name>/1/rrt_gldv2_roxford_r50_gldv2.pt \
   desc_name=<model_name>
```

---

## Self-supervised Pretraining

```
# Pretrained using BCE
python experiment_pretrain.py -F logs/<path>/<model_name> with \
#     dataset.gldv2_r50_self model.RRT max_norm=0.0 loss="bce"

# finetune
python experiment.py -F logs/<path>/<model_name> with \
#     dataset.gldv2_roxford_r50_gldv2 model.RRT max_norm=0.0 \
#     resume=logs/<path>/<model_name>/gldv2_r50_selfBCE_h8/1/rrt_gldv2_r50_self.pt
```