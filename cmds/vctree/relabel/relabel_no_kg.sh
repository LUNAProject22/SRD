MODEL_NAME="vctree"
N=25 # 20, 25, 30, 35, 40, 45, 50
TOP_K_FREQ=$((50-N))
OUTPUT_DIR="output/relabel-$MODEL_NAME/without_idf_N$N"
mkdir -p $OUTPUT_DIR

python maskrcnn_benchmark/data/datasets/vg_new_relation.py \
    --config-file "configs/relation_deduction/relabel_new_relation_pretrain.yaml" \
    NEW_RELATION.REL_CACHE_PREFIX ${MODEL_NAME}_pretrain_N${N}_no_kg \
    NEW_RELATION.OUTPUT_DIR $OUTPUT_DIR \
    USE_LOGITS.DATA_WITH_LOGITS output/relabel-$MODEL_NAME/sup_data_with_${MODEL_NAME}_prediction.pk \
    NEW_RELATION.TOP_K_FREQ $TOP_K_FREQ \
    NEW_RELATION.USE_KG False
