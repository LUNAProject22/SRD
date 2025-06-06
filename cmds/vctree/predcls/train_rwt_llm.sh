# screen -S vctree
# conda activate sgg

MODEL_NAME="vctree"
llm="palm"
N=25
OUTPATH="checkpoints/$MODEL_NAME/$MODEL_NAME-precls-rwt-$llm-N$N"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=1 python tools/relation_train_net.py \
    --config-file "configs/rwt/rwt_predcls.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
    DATASETS.NEW_RELATION.NEW_TRIPLE_FILE output/relabel-$MODEL_NAME/$llm/N$N/relation_cache_${MODEL_NAME}_pretrain_tfidf_${llm}_N${N}_CD_train.npy \
    OUTPUT_DIR $OUTPATH
