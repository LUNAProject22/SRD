import os
import torch
import numpy as np
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from maskrcnn_benchmark.data.datasets.visual_genome import load_info
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, \
    SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall


ind_to_classes, ind_to_predicates, _ = load_info(dict_file="datasets/vg/VG-SGG-dicts-with-attri.json")
predicate_to_ind = {p: i for i, p in enumerate(ind_to_predicates)}

freq_rels = [0, 31, 20, 22, 30, 48, 29, 50, 1, 21, 8, 43, 40, 49, 41, 23,
             38, 6, 7, 33, 11, 46, 16, 25, 47, 19, 5, 9, 35, 24, 10, 4, 14,
             13, 12, 36, 44, 42, 32, 2, 28, 26, 45, 3, 17, 18, 34, 27, 37, 39, 15]


def exclude_predicate(groundtruth, prediction, id_to_exclude):
    """groundtruth.get_field('relation_tuple'), 
    prediction.get_field('rel_pair_idxs'),
    prediction.get_field('pred_rel_scores')
    prediction.get_field('pred_rel_labels')
    """
    relation_tuple = groundtruth.get_field('relation_tuple')
    rel_pair_idxs = prediction.get_field('rel_pair_idxs')
    pred_rel_scores = prediction.get_field('pred_rel_scores')
    pred_rel_labels = prediction.get_field('pred_rel_labels')

    gt_mask = torch.ones((relation_tuple.shape[0]), dtype=torch.bool) # True
    for i, p in enumerate(relation_tuple[:, 2]):
        if p in id_to_exclude:
            gt_mask[i] = False
    new_relation_tuple = relation_tuple[gt_mask]
    groundtruth.add_field("relation_tuple", torch.LongTensor(new_relation_tuple))

    pred_mask = torch.ones((pred_rel_labels.shape[0]), dtype=torch.bool) # True
    for i, p in enumerate(pred_rel_labels):
        if p in id_to_exclude:
            pred_mask[i] = False
    new_pred_rel_labels = pred_rel_labels[pred_mask]
    new_rel_pair_idxs = rel_pair_idxs[pred_mask]
    new_pred_rel_scores = pred_rel_scores[pred_mask]
    prediction.add_field("rel_pair_idxs", new_rel_pair_idxs)
    prediction.add_field("pred_rel_labels", new_pred_rel_labels)
    prediction.add_field("pred_rel_scores", new_pred_rel_scores)
    return



def do_vg_evaluation(
    cfg,
    groundtruths,
    predictions,
    output_folder,
    logger,
    iou_types=("relations"),
    mode='predcls',
    id_to_exclude=[]
):
    # get zeroshot triplet
    zeroshot_triplet = torch.load("maskrcnn_benchmark/data/datasets/evaluation/vg/zeroshot_triplet.pytorch", map_location=torch.device("cpu")).long().numpy()

    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    ds_name = "vg"

    result_str = '\n' + '=' * 100 + '\n'
    if "bbox" in iou_types and mode == 'sgdet':
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist() # integer
            boxes = gt.bbox.tolist() # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()

        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(ind_to_classes) if name != '__background__'
                ],
            'annotations': anns,
        }

        fauxcoco.createIndex()

        # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert('xywh').bbox.detach().cpu().numpy() # xywh
            score = prediction.get_field('pred_scores').detach().cpu().numpy() # (#objs,)
            label = prediction.get_field('pred_labels').detach().cpu().numpy() # (#objs,)
            # for predcls, we set label and score to groundtruth
            if mode == 'predcls':
                label = prediction.get_field('labels').detach().cpu().numpy()
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            image_id = np.asarray([image_id]*len(box))
            cocolike_predictions.append(
                np.column_stack((image_id, box, score, label))
                )
            # logger.info(cocolike_predictions)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAp = coco_eval.stats[1]

        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += '=' * 100 + '\n'

    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # test on different distribution
        eval_zeroshot_recall = SGZeroShotRecall(result_dict)
        eval_zeroshot_recall.register_container(mode)
        evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

        # test on no graph constraint zero-shot recall
        eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
        eval_ng_zeroshot_recall.register_container(mode)
        evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall

        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, ind_to_predicates, print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        # prepare all inputs
        global_container = {}
        global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes

        for groundtruth, prediction in tqdm(zip(groundtruths, predictions), total=len(groundtruths)):
            # exclude certain predicates from gt and prediction
            exclude_predicate(groundtruth, prediction, id_to_exclude)
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, ds_name)

        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)

        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        result_str += eval_zeroshot_recall.generate_print_string(mode)
        # result_str += eval_ng_zeroshot_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        result_str += eval_ng_mean_recall.generate_print_string(mode)

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'


    logger.info(result_str)

    if "relations" in iou_types:
        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict_exclude.pth'))
        return float(np.mean(result_dict[mode + cfg.GLOBAL_SETTING.CHOOSE_BEST_MODEL_BY_METRIC][100]))
    elif "bbox" in iou_types:
        return float(mAp)
    else:
        return -1



def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, ds_name):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()           # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()          # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()                  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()     # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()              # (#pred_objs, )


    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container, ds_name)
    evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container, ds_name)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')


    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode, ds_name)

    # No Graph Constraint
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode, ds_name)
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode, ds_name)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # No Graph Constraint Mean Recall
    evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode, ds_name)

    return



if __name__=="__main__":
    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file("configs/e2e_relation_X_101_32_8_FPN_1x.yaml") # to modify

    prediction_file = "checkpoints/transformer/transformer-precls/inference/VG_stanford_filtered_with_attribute_test/eval_results.pytorch"
    gt_and_preds = torch.load(prediction_file)
    groundtruths = gt_and_preds["groundtruths"]
    predictions = gt_and_preds["predictions"]

    logger = setup_logger("sgg", ".", 0, filename="log-test-exclude.txt")

    for pred_to_exclude in [5, 10, 15, 20, 25]:
        if type(pred_to_exclude) == int:
            id_to_exclude = freq_rels[:pred_to_exclude+1]
        else:
            id_to_exclude = [predicate_to_ind[p] for p in pred_to_exclude]
        word_to_exclude = [ind_to_predicates[id] for id in id_to_exclude]

        logger.info(f"{prediction_file}\nExclude: {len(word_to_exclude)-1}, {word_to_exclude[1:]}")

        do_vg_evaluation(
            cfg,
            groundtruths,
            predictions,
            output_folder=None,
            logger=logger,
            mode='predcls', # to change
            id_to_exclude=id_to_exclude
        )
