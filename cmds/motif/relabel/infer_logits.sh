MODEL_NAME="motif"
OUTPUT_DIR="output/relabel-$MODEL_NAME"

python tools/relation_relabel.py \
    --config-file "configs/relation_deduction/infer_logits.yaml" \
    OUTPUT_DIR $OUTPUT_DIR \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/motifs-precls/model_best_mr20.pth \
    USE_LOGITS.DATA_WITH_LOGITS $OUTPUT_DIR/sup_data_with_${MODEL_NAME}_prediction.pk
    