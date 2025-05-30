# screen -S motif
# conda activate sgg

MODEL_NAME="motif"
OUTPATH="checkpoints/motifs-precls"

CUDA_VISIBLE_DEVICES=1 python tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/model_0026000.pth \
    OUTPUT_DIR $OUTPATH
