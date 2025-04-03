# screen -S motif
# conda activate sgg

MODEL_NAME="motif"
N=100
OUTPATH="checkpoints/GQA/$MODEL_NAME/$MODEL_NAME-gqa-sgcls-rwt-N$N-dspc-idf"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=1 python tools/relation_train_net.py \
    --config-file "configs/rwt/rwt_gqa.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/GQA/pretrained_faster_rcnn/model_final_from_vg.pth \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    SOLVER.MAX_ITER 40000 SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 \
    DATASETS.NEW_RELATION.NEW_TRIPLE_FILE output/GQA/relabel-$MODEL_NAME/N$N/relation_cache_${MODEL_NAME}_pretrain_idf_N${N}_CD.npy \
    OUTPUT_DIR $OUTPATH
