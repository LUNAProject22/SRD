# screen -S trans
# conda activate sgg

OUTPATH="checkpoints/transformer/transformer-precls"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=1 python tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    DATASETS.NEW_RELATION.ENABLE False DATASETS.NEW_RELATION.NEW_TRIPLE_FILE "" \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
    SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR $OUTPATH