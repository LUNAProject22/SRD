# screen -S trans
# conda activate sgg

MODEL_NAME="motif"
OUTPATH="checkpoints/$MODEL_NAME/$MODEL_NAME-precls"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=3 python tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    DATASETS.NEW_RELATION.ENABLE False DATASETS.NEW_RELATION.NEW_TRIPLE_FILE "" \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
    OUTPUT_DIR $OUTPATH
