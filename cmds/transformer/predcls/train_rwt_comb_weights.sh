# screen -S trans
# conda activate sgg

MODEL_NAME="transformer"
N=25
OUTPATH="checkpoints/$MODEL_NAME/$MODEL_NAME-precls-rwt-N$N-f15o1-comb_weights"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
    --config-file "configs/rwt/rwt_predcls.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    SOLVER.BASE_LR 0.001 SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    SOLVER.MAX_ITER 16000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 5000 \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
    DATASETS.NEW_RELATION.NEW_TRIPLE_FILE output/relabel-transformer/N25/relation_cache_transformer_pretrain_tfidf_N25_CD_train-min_freq_15-max_occur_1.npy \
    WSUPERVISE.RECALL_FILE output/rwt_weights/recall20_${MODEL_NAME}_baseline.pth \
    OUTPUT_DIR $OUTPATH
