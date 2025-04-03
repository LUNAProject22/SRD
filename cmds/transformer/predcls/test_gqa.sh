# screen -S trans
# conda activate sgg

MODEL_NAME="transformer"
OUTPATH="checkpoints/GQA/$MODEL_NAME/$MODEL_NAME-gqa-precls"

CUDA_VISIBLE_DEVICES=2 python tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_gqa.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/model_best_mean_recall.pth \
    OUTPUT_DIR $OUTPATH
