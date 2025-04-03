# screen -S trans
# conda activate sgg

MODEL_NAME="transformer"
N=25
OUTPATH="checkpoints/$MODEL_NAME/$MODEL_NAME-sgdet-N$N"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=3 python tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
    SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 4000 \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
    DATASETS.NEW_RELATION.NEW_TRIPLE_FILE output/relabel-$MODEL_NAME/N$N/relation_cache_${MODEL_NAME}_pretrain_tfidf_N${N}_CD_train.npy \
    OUTPUT_DIR $OUTPATH
