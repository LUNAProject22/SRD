MODEL_NAME="transformer"
N=50
TOP_K_FREQ=$((100-N))
OUTPUT_DIR="output/GQA/relabel-$MODEL_NAME/N$N"
mkdir -p $OUTPUT_DIR

python maskrcnn_benchmark/data/datasets/gqa_relation_deduction.py \
    --config-file "configs/relation_deduction/relabel_new_relation_pretrain_gqa.yaml" \
    NEW_RELATION.REL_CACHE_PREFIX ${MODEL_NAME}_pretrain_N$N \
    NEW_RELATION.OUTPUT_DIR $OUTPUT_DIR \
    USE_LOGITS.DATA_WITH_LOGITS output/GQA/relabel-$MODEL_NAME/sup_data_with_${MODEL_NAME}_prediction.pk \
    USE_LOGITS.LOGITS_COMB_PRIOR $OUTPUT_DIR/sup_data_with_${MODEL_NAME}_pred_comb_N${N}_CD.pk \
    NEW_RELATION.USE_CONTEXT True \
    NEW_RELATION.USE_DISTILLATION True \
    NEW_RELATION.TOP_K_FREQ $TOP_K_FREQ \
    NEW_RELATION.CLUSTER_TO_KNOWLEDGE triple_prob-freq_matrix-F1-N$N.pkl
