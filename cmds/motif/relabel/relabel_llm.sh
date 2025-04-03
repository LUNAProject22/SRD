MODEL_NAME="motif"
llm="palm"
N=25 # 20, 25, 30, 35, 40, 45, 50
TOP_K_FREQ=$((50-N))
OUTPUT_DIR="output/relabel-$MODEL_NAME/${llm}/N$N"
mkdir -p $OUTPUT_DIR

python maskrcnn_benchmark/data/datasets/vg_new_relation.py \
    --config-file "configs/relation_deduction/relabel_new_relation_pretrain.yaml" \
    NEW_RELATION.REL_CACHE_PREFIX ${MODEL_NAME}_pretrain_tfidf_${llm}_N$N \
    NEW_RELATION.OUTPUT_DIR $OUTPUT_DIR \
    USE_LOGITS.DATA_WITH_LOGITS output/relabel-$MODEL_NAME-i1/sup_data_with_${MODEL_NAME}_prediction.pk \
    USE_LOGITS.LOGITS_COMB_PRIOR $OUTPUT_DIR/sup_data_with_${MODEL_NAME}_pred_comb_tfidf_${llm}_N${N}_CD.pk \
    NEW_RELATION.TOP_K_FREQ $TOP_K_FREQ \
    NEW_RELATION.CLUSTER_TO_KNOWLEDGE triple_prob-context-distill-tfidf-masked_new_rels-${llm}_prompt4-N$N.pkl \
    NEW_RELATION.N_SPO_RULE False
