# screen -S vctree
# conda activate sgg

MODEL_NAME="vctree"
N=35
OUTPATH="checkpoints/$MODEL_NAME/$MODEL_NAME-sgcls-N$N"
mkdir -p $OUTPATH

CUDA_VISIBLE_DEVICES=2 python tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
    SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 2 DTYPE "float16" \
    MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth \
    DATASETS.NEW_RELATION.NEW_TRIPLE_FILE output/OSP/vctree-i1/relation_cache_vctree-i1_pretrain_tfidf_N35_CD_train.npy \
    OUTPUT_DIR $OUTPATH
