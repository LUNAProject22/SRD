# screen -S trans
# conda activate sgg

MODEL_NAME="transformer"
OUTPATH="checkpoints/GQA/$MODEL_NAME/$MODEL_NAME-gqa-precls"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=1 python tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_gqa.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/GQA/pretrained_faster_rcnn/model_final_from_vg.pth \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    DATASETS.NEW_RELATION.ENABLE False DATASETS.NEW_RELATION.NEW_TRIPLE_FILE "" \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    SOLVER.MAX_ITER 40000 SOLVER.CHECKPOINT_PERIOD 10000 \
    OUTPUT_DIR $OUTPATH