# Effective Scene Graph Generation by Statistical Relation Distillation

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/LUNAProject22/SRD/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8.x-%237732a8)](https://pytorch.org/get-started/previous-versions/)

Our paper [Effective Scene Graph Generation by Statistical Relation Distillation](https://openaccess.thecvf.com/content/WACV2025/html/Nguyen_Effective_Scene_Graph_Generation_by_Statistical_Relation_Distillation_WACV_2025_paper.html) has been accepted by WACV 2025. This repo is built upon [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). Thanks for their wonderful work.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.


## Pretrained Models
You can download the [pretrained Faster R-CNN](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ) we used in the paper.

After you download the [Faster R-CNN model](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779870&authkey=AH5CPVb9g5E67iQ), please extract all the files to the directory `/home/username/checkpoints/pretrained_faster_rcnn`. To train your own Faster R-CNN model, please follow the next section.

The above pretrained Faster R-CNN model achives 38.52/26.35/28.14 mAp on VG train/val/test set respectively.

## Faster R-CNN pre-training
Open a screen session named "sgg" ```$ screen -S sgg```
The following command can be used to train your own Faster R-CNN model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.STEPS "(30000, 45000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /home/username/checkpoints/pretrained_faster_rcnn SOLVER.PRE_VAL False
```
where ```CUDA_VISIBLE_DEVICES``` and ```--nproc_per_node``` represent the number of GPUs you use, ```--config-file``` means the config we use, where you can change other parameters. ```SOLVER.IMS_PER_BATCH``` and ```TEST.IMS_PER_BATCH``` are the training and testing batch size respectively, ```DTYPE "float16"``` enables Automatic Mixed Precision supported by [APEX](https://github.com/NVIDIA/apex), ```SOLVER.MAX_ITER``` is the maximum iteration, ```SOLVER.STEPS``` is the steps where we decay the learning rate, ```SOLVER.VAL_PERIOD``` and ```SOLVER.CHECKPOINT_PERIOD``` are the periods of conducting val and saving checkpoint, ```MODEL.RELATION_ON``` means turning on the relationship head or not (since this is the pretraining phase for Faster R-CNN only, we turn off the relationship head),  ```OUTPUT_DIR``` is the output directory to save checkpoints and log (considering `/home/username/checkpoints/pretrained_faster_rcnn`), ```SOLVER.PRE_VAL``` means whether we conduct validation before training or not.


## Training on Scene Graph Generation

There are **three standard protocols**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. We use two switches ```MODEL.ROI_RELATION_HEAD.USE_GT_BOX``` and ```MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL``` to select the protocols.

For **Predicate Classification (PredCls)**, we need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Baseline Models
We abstract various SGG models to be different ```relation-head predictors``` in the file ```roi_heads/relation_head/roi_relation_predictors.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor. To select our predefined models, you can use ```MODEL.ROI_RELATION_HEAD.PREDICTOR```.

For [Neural-MOTIFS](https://arxiv.org/abs/1711.06640) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
```
For [VCTree](https://arxiv.org/abs/1812.01880) Model:
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
```
For our predefined Transformer Model (Note that Transformer Model needs to change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000).), which is provided by [Jiaxin Shi](https://github.com/shijx12):
```bash
MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
```

The default settings are under ```configs/e2e_relation_X_101_32_8_FPN_1x.yaml``` and ```maskrcnn_benchmark/config/defaults.py```. The priority is ```command > yaml > defaults.py```

### Customize Your Own Model
If you want to customize your own model, you can refer ```maskrcnn-benchmark/modeling/roi_heads/relation_head/model_XXXXX.py``` and ```maskrcnn-benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py```. You also need to add corresponding nn.Module in ```maskrcnn-benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py```. Sometimes you may also need to change the inputs & outputs of the module through ```maskrcnn-benchmark/modeling/roi_heads/relation_head/relation_head.py```.

### Examples of Commands
Training Example 1: (PreCls, Motif Model)
```bash
bash cmds/motif/predcls/train_baseline.sh
```
Training Example 2: (SGCls, Motif Model, rwt)
```bash
bash cmds/motif/sgcls/train_rwt.sh
```
Relabel script: (Motif Model)
```bash
bash cmds/motif/relabel/relabel.sh
```
Test Example 1 : (PreCls, Transformer Model)
```bash
bash cmds/transformer/predcls/test.sh
```

## Metrics and Results
Explanation of metrics in our toolkit and reported results are given in [METRICS.md](METRICS.md)

| Method         | PredCls                 | SGCls                   | SGDet                    |
|----------------|-------------------------|-------------------------|--------------------------|
|                | mR@20 /50 /100          | mR@20 /50 /100          | mR@20 /50 /100           |
| Motif (Base)   | 12.1 / 15.7 / 17.4      | 7.2 / 8.7 / 9.3         | 5.1 / 6.5 / 7.8          |
|  –Ours         | 31.9 / 37.9 / 40.5      | 18.9 / 21.9 / 22.8      | 13.5 / 17.9 / 20.6       |
| VCTree (Base)  | 12.4 / 15.4 / 16.6      | 6.3 / 7.5 / 8.0         | 4.9 / 6.6 / 7.7          |
|  –Ours         | 33.4 / 39.0 / 41.1      | 23.0 / 26.6 / 27.6      | 12.8 / 16.7 / 19.6       |
| Transf. (Base) | 14.8 / 19.2 / 20.5      | 8.9 / 11.6 / 12.6       | 5.6 / 7.7 / 9.0          |
|  –Ours         | 34.0 / 39.6 / 41.7      | 20.1 / 23.0 / 23.9      | 14.3 / 18.3 / 20.8       |


## Citations

If you find this project helps your research, please kindly consider citing our paper in your publications.

```
@InProceedings{Nguyen_2025_WACV,
    author    = {Nguyen, Thanh-Son and Yang, Hong and Fernando, Basura},
    title     = {Effective Scene Graph Generation by Statistical Relation Distillation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {8409-8419}
}
```
