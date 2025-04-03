# screen -S clip
# conda activate sgg

MODEL_NAME="clip"
OUTPATH="checkpoints/Gx2-RPA-V220-CPT--ViT-B-16-LR1e-06-B60-Pamp_bf16-openai-E3-2C51NE0.2WLoss-2023_11_06-17_32_46"
EPOCH=2

CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CLIPPredictor \
    MODEL.CLIP_CKPT $OUTPATH/checkpoints/epoch_$EPOCH.pt \
    TEST.IMS_PER_BATCH 8 DTYPE "float16" \
    TEST.RELATION.REQUIRE_OVERLAP True \
    TEST.RELATION.TEST_OVERLAP_THRES 0.01 \
    OUTPUT_DIR $OUTPATH