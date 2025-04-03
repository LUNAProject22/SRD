# screen -S trans
# conda activate sgg

MODEL_NAME="transformer"
N=100
OUTPATH="checkpoints/GQA/$MODEL_NAME/$MODEL_NAME-gqa-sgdet-rwt-N$N-dspc-idf"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
    --config-file "configs/rwt/rwt_gqa.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/model_best_mean_recall.pth \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    OUTPUT_DIR $OUTPATH
