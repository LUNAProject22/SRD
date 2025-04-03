MODEL_NAME="transformer"
OUTPUT_DIR="output/GQA/relabel-$MODEL_NAME"
mkdir -p $OUTPATH

python tools/relation_relabel.py \
    --config-file "configs/relation_deduction/infer_logits.yaml" \
    OUTPUT_DIR $OUTPUT_DIR \
    MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/GQA/$MODEL_NAME/$MODEL_NAME-gqa-precls/model_best_mean_recall.pth \
    USE_LOGITS.DATA_WITH_LOGITS $OUTPUT_DIR/sup_data_with_${MODEL_NAME}_prediction.pk
    