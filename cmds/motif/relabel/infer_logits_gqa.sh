MODEL_NAME="motif"
OUTPUT_DIR="output/GQA/relabel-$MODEL_NAME"
mkdir -p $OUTPATH

python tools/relation_relabel.py \
    --config-file "configs/relation_deduction/infer_logits_gqa.yaml" \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/GQA/$MODEL_NAME/$MODEL_NAME-gqa-precls/model_final.pth \
    USE_LOGITS.DATA_WITH_LOGITS $OUTPUT_DIR/sup_data_with_${MODEL_NAME}_prediction.pk \
    USE_LOGITS.STAGE "get-logits" \
    OUTPUT_DIR $OUTPUT_DIR
