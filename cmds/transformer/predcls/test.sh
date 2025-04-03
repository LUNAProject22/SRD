# screen -S trans
# conda activate sgg

MODEL_NAME="transformer"
OUTPATH="checkpoints/transformer/transformer-precls-rwt-N25-f15o1"

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    MODEL.PRETRAINED_DETECTOR_CKPT $OUTPATH/model_final.pth \
    OUTPUT_DIR $OUTPATH
